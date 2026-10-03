/**
 * Feature: spec/features/jev-http-endpoints-router-classifier-systemone-alias-health-openapi-error-envelope.feature
 *
 * This test file validates the acceptance criteria defined in the feature file.
 * Scenarios map directly to Gherkin scenarios.
 *
 * All scenarios are weight-free: the routes are served by ntex's in-process
 * `test::server` (loopback socket) against a mock `Answerer` (JEV-005
 * trait) — no checkpoint, no GPU. The protocol envelope assertions are
 * byte-exact against docs/jev-server/protocol-reference.md "Errors".
 */

use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use laya::schema::Question;
use laya::server::state::{Answerer, ServerConfig, ServerState};
use laya::Answer;
use serde_json::json;
use serde_json::Value;
use ntex::client::ClientResponse;
use ntex::http::{Method, StatusCode};
use ntex::web::test::{self, TestServer};
use ntex::web::App;

const MODEL: &str = "convaiinnovations/laya";

/// Weight-free `Answerer`: records forward calls; optional one-shot failure.
#[derive(Default)]
struct MockAnswerer {
    fail_once: AtomicBool,
    calls: AtomicUsize,
}

impl MockAnswerer {
    fn forwards(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Answerer for MockAnswerer {
    fn checkpoint_limits(&self) -> (usize, usize) {
        (1024, 128)
    }

    fn admit(
        &self,
        _state: &serde_json::Value,
        _questions: &[(String, Question)],
        cap: usize,
    ) -> Result<Vec<usize>, laya::server::state::AdmissionError> {
        Ok(vec![16.min(cap)])
    }

    fn system_one(
        &self,
        _state: &serde_json::Value,
        questions: &[(String, Question)],
    ) -> anyhow::Result<Vec<(String, Answer)>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_once.swap(false, Ordering::SeqCst) {
            anyhow::bail!("simulated internal forward error");
        }
        Ok(questions
            .iter()
            .map(|(qid, _)| (qid.clone(), Answer::Noul { noul: 0.5, act_probability: 0.0 }))
            .collect())
    }
}

async fn server_with(mock: Arc<MockAnswerer>) -> TestServer {
    let state = ServerState::new(mock, MODEL, ServerConfig::default());
    test::server(async move || App::new().configure(laya::server::router::routes(state.clone()))).await
}

/// One request against the test server.
async fn req(app: &TestServer, method: &str, uri: &str, body: Option<&str>, content_type: Option<&str>) -> ClientResponse {
    let mut request = app.request(method.parse::<Method>().unwrap(), app.url(uri));
    if let Some(ct) = content_type {
        request = request.header("content-type", ct);
    }
    request.send_body(body.unwrap_or_default().to_owned()).await.unwrap()
}

/// The (status, body-text) of a response, consuming the body.
async fn take(res: ClientResponse) -> (StatusCode, String) {
    let status = res.status();
    let bytes = res.body().limit(usize::MAX).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

/// A valid mixed request body: one choice, one score, one noul question.
fn mixed_body() -> String {
    json!({
        "model": MODEL,
        "state": "We were billed twice for March.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should own this?",
                "criteria": { "billing": null, "technical": null }
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent?",
                "criteria": ["Routine", "Important", "Critical"]
            },
            "refund": {
                "type": "noul",
                "instructions": "Should we refund?",
                "criteria": { "true": "Yes", "false": "No" }
            }
        }
    })
    .to_string()
}

/// The protocol 422 envelope: `{"error":{message,type,code,param,details[]}}`.
fn assert_422_envelope(status: &StatusCode, body: &str, detail_hint: &str) {
    assert_eq!(*status, StatusCode::UNPROCESSABLE_ENTITY, "expected 422, got {status}: {body}");
    let v: Value = serde_json::from_str(body).unwrap_or_else(|e| panic!("422 body must be JSON: {e}"));
    assert_eq!(v["error"]["type"], "invalid_request_error", "envelope type: {body}");
    assert_eq!(v["error"]["code"], 422, "envelope code: {body}");
    assert!(
        v["error"]["message"].as_str().is_some_and(|m| m.contains(detail_hint)),
        "422 message must name the problem ({detail_hint:?}): {body}"
    );
    assert!(v["error"]["details"].is_array(), "envelope must carry a details array: {body}");
}

/// Scenario: The systemone alias behaves identically to classifier
#[ntex::test]
async fn the_systemone_alias_behaves_identically_to_classifier() {
    // @step Given the server is running with model "convaiinnovations/laya"
    let mock = Arc::new(MockAnswerer::default());
    let app = server_with(mock.clone()).await;
    let body = mixed_body();

    // @step When the client POSTs a mixed choice/score/noul body to /v1/systemone
    let (alias_status, alias_body) =
        take(req(&app, "POST", "/v1/systemone", Some(&body), Some("application/json")).await).await;
    let (main_status, main_body) =
        take(req(&app, "POST", "/v1/classifier", Some(&body), Some("application/json")).await).await;

    // @step Then the response's answers and usage are byte-equivalent in shape to the /v1/classifier response for the same body
    assert_eq!(alias_status, StatusCode::OK, "the alias must serve the request: {alias_body}");
    assert_eq!(main_status, StatusCode::OK, "classifier must serve the request: {main_body}");
    assert!(main_body.eq(&alias_body), "byte-equivalent bodies: main={main_body} alias={alias_body}");
    let parsed: Value = serde_json::from_str(&main_body).unwrap();
    assert_eq!(parsed["model"], MODEL);
    for qid in ["route", "urgency", "refund"] {
        assert!(parsed["answers"].get(qid).is_some(), "answers must name the question id {qid}: {main_body}");
    }
    assert_eq!(parsed["answers"]["refund"]["type"], "noul", "each answer must carry its protocol type");
    assert_eq!(parsed["usage"]["output_tokens"], 0, "usage.output_tokens must be 0");
}

/// Scenario: Health reports readiness without inference
#[ntex::test]
async fn health_reports_readiness_without_inference() {
    // @step Given the server is running with model "convaiinnovations/laya"
    let mock = Arc::new(MockAnswerer::default());
    let app = server_with(mock.clone()).await;

    // @step When the client GETs /health
    let (status, body) = take(req(&app, "GET", "/health", None, None).await).await;

    // @step Then the response is 200 with body {"status":"ready","model":"convaiinnovations/laya"}
    assert_eq!(status, StatusCode::OK, "health body: {body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["status"], "ready");
    assert_eq!(v["model"], MODEL);

    // @step And no forward pass was executed (assertable with an instrumented answerer)
    assert_eq!(mock.forwards(), 0, "health must not trigger a forward pass");
}

/// Scenario: Oversized bodies and wrong content types are rejected before parsing
#[ntex::test]
async fn oversized_bodies_and_wrong_content_types_are_rejected_before_parsing() {
    // @step Given the server is running
    let mock = Arc::new(MockAnswerer::default());
    let app = server_with(mock.clone()).await;

    // @step When the client POSTs a 2 MiB JSON body to /v1/classifier, or a valid JSON body with Content-Type: text/plain
    // 2 MiB (1048576) body: exceeds the 1 MiB limit by one byte.
    let big = json!({ "model": MODEL, "state": "x".repeat(2 * 1024 * 1024), "questions": {} }).to_string();
    let (big_status, big_body) =
        take(req(&app, "POST", "/v1/classifier", Some(&big), Some("application/json")).await).await;

    let valid_body = mixed_body();
    let (ct_status, ct_body) =
        take(req(&app, "POST", "/v1/classifier", Some(&valid_body), Some("text/plain")).await).await;

    // @step Then both requests receive the 422 error envelope and no JSON parsing or inference occurred
    assert_422_envelope(&big_status, &big_body, "1 MiB");
    assert_422_envelope(&ct_status, &ct_body, "application/json");
    assert_eq!(mock.forwards(), 0, "rejected requests must never reach inference");
}

/// Scenario: Model identity mismatch is a 422 with the loaded model named
#[ntex::test]
async fn model_identity_mismatch_is_a_422_with_the_loaded_model_named() {
    // @step Given the server loaded model "convaiinnovations/laya"
    let mock = Arc::new(MockAnswerer::default());
    let app = server_with(mock.clone()).await;

    // @step When the client POSTs an otherwise-valid body whose model field is "other"
    let mut wrong_model: Value = serde_json::from_str(&mixed_body()).unwrap();
    wrong_model["model"] = json!("other");
    let (status, body) =
        take(req(&app, "POST", "/v1/classifier", Some(&wrong_model.to_string()), Some("application/json")).await).await;

    // @step Then the response is 422 with error.message "Loaded model is 'convaiinnovations/laya'"
    assert_422_envelope(&status, &body, "Loaded model is 'convaiinnovations/laya'");
    assert_eq!(mock.forwards(), 0, "a model mismatch must be rejected before inference");
}

/// Scenario: Internal forward errors surface as 500 and the server keeps serving
#[ntex::test]
async fn internal_forward_errors_surface_as_500_and_the_server_keeps_serving() {
    // @step Given the answerer is configured to raise an internal error on the next forward
    let mock = Arc::new(MockAnswerer { fail_once: AtomicBool::new(true), ..Default::default() });
    let app = server_with(mock.clone()).await;

    // @step When the client POSTs a valid body to /v1/classifier
    let body = mixed_body();
    let (status, got) = take(req(&app, "POST", "/v1/classifier", Some(&body), Some("application/json")).await).await;

    // @step Then the response is 500 with body {"detail":"internal error"}
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "got {got}");
    let v: Value = serde_json::from_str(&got).unwrap();
    assert_eq!(v["detail"], "internal error", "the 500 short form: {got}");

    // @step And a subsequent valid request succeeds (the server stays up)
    let (status2, body2) = take(req(&app, "POST", "/v1/classifier", Some(&body), Some("application/json")).await).await;
    assert_eq!(status2, StatusCode::OK, "the server must keep serving: {body2}");
    assert_eq!(mock.forwards(), 2, "exactly the two non-cancelled requests may forward");
}

/// Scenario: The OpenAPI document is served for discovery
#[ntex::test]
async fn the_openapi_document_is_served_for_discovery() {
    // @step Given the server is running
    let mock = Arc::new(MockAnswerer::default());
    let app = server_with(mock).await;

    // @step When the client GETs /openapi.json
    let (status, body) = take(req(&app, "GET", "/openapi.json", None, None).await).await;

    // @step Then the response is 200 with a static OpenAPI 3.1 document describing /v1/classifier, /v1/systemone, and /health
    assert_eq!(status, StatusCode::OK, "openapi body: {body}");
    let v: Value = serde_json::from_str(&body).expect("openapi.json must be valid JSON");
    assert!(
        v["openapi"].as_str().is_some_and(|s| s.starts_with("3.1")),
        "must be OpenAPI 3.1: {v}"
    );
    for path in ["/v1/classifier", "/v1/systemone", "/health"] {
        assert!(v["paths"].get(path).is_some(), "paths must describe {path}: {v}");
    }
}
