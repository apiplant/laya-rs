//! Server lifecycle (JEV-006): [`start_server`] / [`ServerHandle`].
//!
//! Build the [`ServerState`] over the given [`Answerer`] seam, bind the TCP
//! listener (port 0 = an ephemeral port) and run an ntex `HttpServer` over the
//! JEV-004 routes. Must be called from inside an ntex runtime (`#[ntex::main]`,
//! `System::new(..).block_on(..)` or `#[ntex::test]`).
//!
//! ntex owns shutdown: SIGINT/SIGTERM (or [`ServerHandle::stop`]) stops the
//! listener, lets in-flight requests finish — an in-flight forward runs to
//! completion under the model lock, it "cannot be interrupted" (reference
//! behavior) — and resolves [`ServerHandle::wait`].
//!
//! Bind failure is an `anyhow` error returned before any HTTP is served; the
//! startup banner is printed by the CLI (`main.rs`), not here.

use std::net::TcpListener;
use std::sync::Arc;

use ntex::server::Server;
use ntex::web::{App, HttpServer};

use crate::server::router::routes;
use crate::server::state::{Answerer, ServerConfig, ServerState};

/// A live server (JEV-006): the actually-bound port, the model identity, and
/// the ntex server handle.
pub struct ServerHandle {
    pub port: u16,
    pub model_name: String,
    server: Server,
}

impl ServerHandle {
    /// Resolve once the server has stopped (after SIGINT/SIGTERM or [`Self::stop`]).
    pub async fn wait(self) {
        let _ = self.server.await;
    }

    /// Trigger a graceful shutdown: stop accepting new connections, let any
    /// in-flight forward complete, and wait for the workers to exit.
    pub async fn stop(self) {
        self.server.stop(true).await;
    }
}

/// Start the Jev-protocol server on `host:port` (port 0 = an ephemeral port).
///
/// The checkpoint (the answerer's backing model) is already loaded by the
/// caller (the CLI loads it first, with stderr progress); this function builds
/// the state, binds, and serves. Returns the [`ServerHandle`] (with the
/// actual bound port); a bind failure is an `anyhow` error before any HTTP is
/// served.
pub async fn start_server(
    host: &str,
    port: u16,
    answerer: Arc<dyn Answerer>,
    model_name: impl Into<String>,
    config: ServerConfig,
) -> anyhow::Result<ServerHandle> {
    let model_name = model_name.into();
    let state = ServerState::new(answerer, model_name.clone(), config);

    let listener = TcpListener::bind((host, port)).map_err(|e| anyhow::anyhow!("failed to bind {host}:{port}: {e}"))?;
    let actual_port = listener
        .local_addr()
        .map_err(|e| anyhow::anyhow!("listener has no local address: {e}"))?
        .port();

    // Inference is serial behind the model lock, so a couple of workers are
    // plenty for connection handling; they all share this one `ServerState`.
    let server = HttpServer::new(async move || App::new().configure(routes(state.clone())))
        .workers(2)
        .listen(listener)?
        .run();

    Ok(ServerHandle { port: actual_port, model_name, server })
}
