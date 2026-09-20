//! `std::time::Instant::now()` panics on `wasm32-unknown-unknown` ("time not
//! implemented on this platform" — there's no OS clock on that target). The
//! per-op `LAYA_TIMING` instrumentation scattered through `modernbert.rs`,
//! `decision_model.rs` and `agent.rs` is native-only diagnostics that's
//! supposed to be a no-op unless that env var is set, but the `Instant::now()`
//! calls themselves were unconditional — so every forward pass panicked in
//! the browser regardless. Swapping each call site to this shim (a real
//! `Instant` natively, an inert dummy on wasm32) fixes that without gating
//! every individual call site by hand.

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
pub struct Instant;

#[cfg(target_arch = "wasm32")]
impl Instant {
    pub fn now() -> Self {
        Instant
    }

    pub fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}
