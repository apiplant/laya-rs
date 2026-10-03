//! The classifier handler — the single handler behind both
//! `POST /v1/classifier` and its exact alias `POST /v1/systemone` (JEV-004).
//!
//! Pipeline (docs/jev-server/JEV-004-http-endpoints.md):
//! 1. Body guard: `DefaultBodyLimit::max(1 MiB)` on the router — an
//!    over-limit body rejects the `Bytes` extractor, mapped to the 422
//!    envelope (no JSON parsing happens).
//! 2. Content-Type guard: missing or `application/json` (any parameters) is
//!    tolerated; anything else is a 422 before parsing.
//! 3. JEV-003 strict parse + validate → 422 envelope.
//! 4. Model identity: `request.model == state.model_name()`, else 422
//!    `Loaded model is '<name>'`.
//! 5. JEV-005 execution seam: admission queue (429), serial forward (500 on
//!    internal error), protocol response (200).
//!
//! The handler never panics on malformed input: every failure path yields a
//! readable protocol status.

use ntex::http::header;
use ntex::util::BytesMut;
use ntex::web::types::{Payload, State};
use ntex::web::{HttpRequest, HttpResponse};

use crate::server::request::{canonical, parse_classifier_request, ClassifierRequest, Context, RequestError};
use crate::server::router::MAX_BODY_BYTES;
use crate::server::state::{ExecutionError, ServerState};

use super::error;

/// How much of a rejected (oversized) body is read and discarded before replying.
const DRAIN_LIMIT: usize = 16 * MAX_BODY_BYTES;

/// Discard the rest of a body we are rejecting, so the client gets our 422 instead of a connection
/// reset mid-upload. Nothing is buffered, and we give up after [`DRAIN_LIMIT`] bytes.
async fn drain(mut payload: Payload) {
    let mut seen = 0;
    while let Some(Ok(chunk)) = payload.recv().await {
        seen += chunk.len();
        if seen > DRAIN_LIMIT {
            break;
        }
    }
}

/// One classifier request (shared by `/v1/classifier` and `/v1/systemone`).
pub async fn classify(state: State<ServerState>, req: HttpRequest, mut payload: Payload) -> HttpResponse {
    // 1. Body guard: over-limit bodies never reach JSON parsing. The declared
    //    Content-Length is checked up front; chunked bodies are cut off as
    //    soon as the running total crosses the limit.
    let too_large = || {
        error::validation_422(vec![RequestError {
            param: "body".to_string(),
            message: "request body exceeds the 1 MiB limit".to_string(),
            type_: "body_limit".to_string(),
        }])
    };
    let declared = req
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok());
    if declared.is_some_and(|n| n > MAX_BODY_BYTES) {
        drain(payload).await;
        return too_large();
    }
    let mut buf = BytesMut::new();
    while let Some(chunk) = payload.recv().await {
        let Ok(chunk) = chunk else { return too_large() };
        if buf.len() + chunk.len() > MAX_BODY_BYTES {
            drain(payload).await;
            return too_large();
        }
        buf.extend_from_slice(&chunk);
    }
    let bytes = buf.freeze();

    // 2. Content-Type guard: JSON only (missing tolerated — clients probing
    //    without a content type still get a readable 422).
    if let Some(ct) = req.headers().get(header::CONTENT_TYPE) {
        let ct = ct.to_str().unwrap_or_default();
        let mime = ct.split(';').next().unwrap_or_default().trim();
        if mime != "application/json" && !mime.ends_with("+json") {
            return error::validation_422(vec![RequestError {
                param: "Content-Type".to_string(),
                message: format!("Content-Type must be application/json (got '{mime}')"),
                type_: "unsupported_content_type".to_string(),
            }])
        }
    }

    // 3. Strict parse + validate (JEV-003).
    let request = match parse_classifier_request(&bytes) {
        Ok(r) => r,
        Err(errors) => return error::validation_422(errors),
    };

    // 4. Model identity (the laya backend is loaded with exactly one model).
    if request.model != state.model_name() {
        return error::validation_422(vec![RequestError {
            param: "model".to_string(),
            message: format!("Loaded model is '{}'", state.model_name()),
            type_: "unknown_model".to_string(),
        }])
    }

    // 5. Execute (JEV-005): admission queue → serial forward → response.
    let state_value = context_to_value(&request);
    match state.execute(state_value, request.questions).await {
        Ok(response) => {
            HttpResponse::Ok().content_type("application/json").body(response.to_json())
        }
        Err(ExecutionError::QueueFull) => error::queue_full_429(),
        Err(ExecutionError::Admission(e)) => error::validation_422(vec![RequestError {
            param: "questions".to_string(),
            message: e.message(),
            type_: "admission".to_string(),
        }]),
        Err(ExecutionError::Internal(_)) => error::internal_500(),
    }
}

/// The request context as the `Value` the execution seam consumes: `state`
/// passes through; `messages` becomes a JSON list of `{role, content}`
/// objects (the reference's serialization for the laya backend —
/// architecture.md pipeline step 5).
fn context_to_value(request: &ClassifierRequest) -> serde_json::Value {
    match &request.context {
        Context::State(state) => state.clone(),
        Context::Messages(messages) => {
            let items = messages
                .iter()
                .map(|m| {
                    let role = match m.role {
                        crate::server::request::Role::System => "system",
                        crate::server::request::Role::Developer => "developer",
                        crate::server::request::Role::User => "user",
                        crate::server::request::Role::Assistant => "assistant",
                    };
                    format!(
                        "{{\"role\":{},\"content\":{}}}",
                        canonical(&serde_json::Value::String(role.to_string())),
                        canonical(&serde_json::Value::String(m.content.clone()))
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            // The parse already validated the entries (string content), so
            // this reconstruction cannot fail.
            serde_json::from_str(&format!("[{items}]")).expect("validated messages serialize")
        }
    }
}
