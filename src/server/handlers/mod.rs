//! Endpoint handlers (JEV-004): `classifier` (the shared
//! `/v1/classifier` + `/v1/systemone` handler), `health`, `openapi`, and
//! `error` (the protocol error-envelope builders, one shared place).

pub mod classifier;
pub mod error;
pub mod health;
pub mod openapi;
