//! `GET /health` (JEV-004): readiness without inference.
//!
//! `{"status":"ready","model":"<loaded model>"}` — 200. The server only
//! listens after the model is loaded (JEV-006), so there is no "loading"
//! state to report; the endpoint is pure state read, never a forward pass.

use ntex::web::types::State;
use ntex::web::HttpResponse;

use crate::server::state::ServerState;

pub async fn health(state: State<ServerState>) -> HttpResponse {
    let body = format!(
        "{{\"status\":\"ready\",\"model\":{}}}",
        serde_json::to_string(state.model_name()).expect("string serializes")
    );
    HttpResponse::Ok().content_type("application/json").body(body)
}
