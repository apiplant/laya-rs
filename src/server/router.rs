//! The Jev-protocol routes (JEV-004): `POST /v1/classifier`, its exact alias
//! `POST /v1/systemone` (same handler, identical behavior), `GET /health`, and
//! `GET /openapi.json`, as an ntex [`ServiceConfig`] callback.
//!
//! [`routes`] is socket-free and test-injectable: `App::new().configure(routes(state))`
//! can be driven with `ntex::web::test::init_service`, or served by
//! `start_server` (JEV-006). The 1 MiB request cap ([`MAX_BODY_BYTES`]) is
//! enforced by the classifier handler itself so an over-limit body yields the
//! protocol 422 envelope rather than the framework's own rejection.

use ntex::web::{self, ServiceConfig};

use crate::server::state::ServerState;

/// The protocol's request-body cap (the reference server's 1 MiB).
pub const MAX_BODY_BYTES: usize = 1_048_576;

/// Register the Jev-protocol routes (and the shared state) on an app.
pub fn routes(state: ServerState) -> impl FnOnce(&mut ServiceConfig) {
    move |cfg| {
        cfg.state(state)
            .route("/v1/classifier", web::post().to(super::handlers::classifier::classify))
            .route("/v1/systemone", web::post().to(super::handlers::classifier::classify)) // exact alias
            .route("/health", web::get().to(super::handlers::health::health))
            .route("/openapi.json", web::get().to(super::handlers::openapi::openapi));
    }
}
