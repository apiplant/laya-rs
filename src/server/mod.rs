//! Jev/Simple-Jev protocol HTTP server (native-only).
//!
//! Serves a `convaiinnovations/laya` checkpoint behind the open Jev
//! classifier contract (`POST /v1/classifier`, alias `/v1/systemone`,
//! `GET /health`) — see `docs/jev-server/architecture.md` for the
//! overarching design and `docs/jev-server/protocol-reference.md` for the
//! v1 protocol contract.
//!
//! Modules: `request` (strict ClassifierRequest parse + validate, JEV-003),
//! `response` (laya-native answer mapping + usage + response assembly,
//! JEV-005), `state` (Answerer seam, pre-inference admission, serial
//! execution + 429 admission queue, JEV-005), `handlers` (the endpoint
//! handlers + protocol error envelopes, JEV-004), `router` (the ntex routes
//! factory, JEV-004), and `server` (the `start_server`/`ServerHandle`
//! lifecycle, JEV-006).

pub mod handlers;
pub mod request;
pub mod response;
pub mod router;
// The `server::server` inception matches the JEV-002 architecture's
// `server/server.rs` layout (the start_server/ServerHandle home, JEV-006).
#[allow(clippy::module_inception)]
pub mod server;
pub mod state;

pub use request::{
    canonical, ChatMessage, ClassifierRequest, Context, parse_classifier_request, RequestError,
    Role,
};
pub use response::{answer_to_jev_json, ClassifierResponse, Usage};
pub use router::routes;
pub use server::{start_server, ServerHandle};
pub use state::{
    admit_questions, AdmissionError, Answerer, ExecutionError, RealAnswerer, ServerConfig,
    ServerState,
};
