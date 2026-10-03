/**
 * Feature: spec/features/jev-protocol-conformance-test-suite-mirroring-the-reference-tests.feature
 *
 * This test file validates the acceptance criteria defined in the feature file.
 * Scenarios map directly to Gherkin scenarios.
 *
 * HTTP-level conformance suite pinning laya-rs to the Jev/Simple-Jev protocol
 * contract, mirroring simple-jev's own test_laya.py (contract) + test_api.py
 * (validation) suites - the jev-server epic's quality gate (JEV-007).
 *
 * Two tiers (the repo's LAYA_TEST_MODEL gating convention):
 * - Weight-free (plain `cargo test`, no checkpoint, no GPU): the routes
 *   served by ntex's in-process `test::server` (loopback socket) against a mock
 *   `Answerer` - the validation matrix, alias, health, openapi shape, the 429
 *   admission queue (a blocking mock forward fills the queue), and 500
 *   recovery.
 * - Model-gated (`LAYA_TEST_MODEL`): `start_server` with a real `RLAgent` on
 *   an ephemeral port, driven by bare-`TcpStream` HTTP round-trips (the
 *   dev-deps carry no HTTP client - the tests/serve_cli.rs pattern). Covers
 *   the contract round-trips (mixed choice/score/noul, wrong model, context
 *   overflow, chat history, determinism) and numerical sanity. Skipped (not
 *   failed) when the env var is unset.
 *
 * Fixtures: JSON files under tests/jev_conformance/ (mixed_fixture.json
 * reuses the README's refund example).
 */

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use std::rc::Rc;

use laya::schema::{QType, Question};
use laya::server::router::routes;
use laya::server::state::{AdmissionError, Answerer, RealAnswerer, ServerConfig, ServerState};
use laya::server::{start_server, ServerHandle};
use laya::{Answer, RLAgent};
use ntex::client::ClientResponse;
use ntex::http::header::{self, HeaderMap};
use ntex::http::{Method, StatusCode};
use ntex::web::test::{self, TestServer};
use ntex::web::App;
use serde_json::{json, Value};

/// The weight-free tier's server: the Jev routes over `state` on an
/// in-process ntex test server.
async fn build_router(state: ServerState) -> TestServer {
    test::server(async move || App::new().configure(routes(state.clone()))).await
}

const MODEL: &str = "convaiinnovations/laya";

// ---------------------------------------------------------------------------
// Weight-free tier: the mock answerer + the oneshot request helpers
// ---------------------------------------------------------------------------

/// A gate a mock forward can park on (fills the admission queue for the 429
/// scenario). `system_one` runs on the blocking pool, so a std condvar park
/// there is safe and does not touch the runtime.
#[derive(Default)]
struct ForwardGate {
    started: AtomicBool,
    released: Mutex<bool>,
    cv: Condvar,
}

impl ForwardGate {
    fn enter(&self) {
        self.started.store(true, Ordering::SeqCst);
        let mut released = self.released.lock().expect("gate lock not poisoned");
        while !*released {
            released = self.cv.wait(released).expect("gate wait not poisoned");
        }
    }

    fn release(&self) {
        let mut released = self.released.lock().expect("gate lock not poisoned");
        *released = true;
        self.cv.notify_all();
    }

    fn started(&self) -> bool {
        self.started.load(Ordering::SeqCst)
    }
}

/// Strictly decreasing distribution summing to 1: weights n, n-1, ..., 1.
/// The first (request-order-first) candidate always wins - deterministic
/// canned answers with a clear argmax.
fn decreasing(n: usize) -> Vec<f32> {
    let weights: Vec<f32> = (1..=n as u32).map(|w| w as f32).rev().collect();
    let sum: f32 = weights.iter().sum();
    weights.into_iter().map(|w| w / sum).collect()
}

/// The mock's canned protocol answer for one question (choice: first
/// candidate wins; score: expected index of the decreasing distribution;
/// noul: a fixed 0.75).
fn canned_answer(q: &Question) -> Answer {
    match q.qtype {
        QType::Choice => {
            let probs = decreasing(q.choice_criteria.len());
            let confidence = probs.first().copied().unwrap_or(0.0);
            let choice = q.choice_criteria.first().map(|(k, _)| k.clone()).unwrap_or_default();
            Answer::Choice {
                choice,
                probabilities: q.choice_criteria.iter().map(|(k, _)| k.clone()).zip(probs).collect(),
                confidence,
                act_probability: 0.0,
            }
        }
        QType::Score => {
            let probs = decreasing(q.score_criteria.len());
            let confidence = probs.first().copied().unwrap_or(0.0);
            let score: f32 = probs.iter().enumerate().map(|(i, p)| i as f32 * p).sum();
            Answer::Score {
                score,
                legend: q.score_criteria.clone(),
                probabilities: probs,
                confidence,
                act_probability: 0.0,
            }
        }
        QType::Noul => Answer::Noul { noul: 0.75, act_probability: 0.0 },
    }
}

/// Weight-free [`Answerer`]: records forward calls, optionally fails the next
/// forward once (500 recovery), optionally rejects admission (overflow 422
/// without inference), and optionally parks its forward on a gate (429 queue).
#[derive(Default)]
struct MockAnswerer {
    calls: AtomicUsize,
    fail_once: AtomicBool,
    admit_error: Mutex<Option<AdmissionError>>,
    gate: Option<Arc<ForwardGate>>,
}

impl MockAnswerer {
    fn forwards(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn with_gate(gate: Arc<ForwardGate>) -> Arc<Self> {
        Arc::new(Self { gate: Some(gate), ..Default::default() })
    }

    fn with_admit_error(e: AdmissionError) -> Arc<Self> {
        Arc::new(Self { admit_error: Mutex::new(Some(e)), ..Default::default() })
    }
}

impl Answerer for MockAnswerer {
    fn checkpoint_limits(&self) -> (usize, usize) {
        (1024, 128)
    }

    fn admit(
        &self,
        _state: &Value,
        questions: &[(String, Question)],
        _cap: usize,
    ) -> Result<Vec<usize>, AdmissionError> {
        if let Some(e) = self.admit_error.lock().expect("admit_error not poisoned").clone() {
            return Err(e);
        }
        Ok(vec![16; questions.len()])
    }

    fn system_one(&self, _state: &Value, questions: &[(String, Question)]) -> anyhow::Result<Vec<(String, Answer)>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_once.swap(false, Ordering::SeqCst) {
            anyhow::bail!("simulated internal forward error");
        }
        if let Some(gate) = &self.gate {
            gate.enter();
        }
        Ok(questions.iter().map(|(qid, q)| (qid.clone(), canned_answer(q))).collect())
    }
}

/// One request against the test server: returns the (status, body) of the response.
async fn hit(
    app: &TestServer,
    method: &str,
    uri: &str,
    body: Option<&str>,
    content_type: Option<&str>,
) -> (StatusCode, String) {
    let (status, body, _headers) = hit_full(app, method, uri, body, content_type).await;
    (status, body)
}

/// Like [`hit`] but also returns the response headers (the 429 Retry-After).
async fn hit_full(
    app: &TestServer,
    method: &str,
    uri: &str,
    body: Option<&str>,
    content_type: Option<&str>,
) -> (StatusCode, String, HeaderMap) {
    let mut request = app.request(method.parse::<Method>().unwrap(), app.url(uri));
    if let Some(ct) = content_type {
        request = request.header("content-type", ct);
    }
    let res: ClientResponse = request.send_body(body.unwrap_or_default().to_owned()).await.expect("server responds");
    let status = res.status();
    let headers = res.headers().clone();
    let bytes = res.body().limit(usize::MAX).await.expect("read body");
    (status, String::from_utf8(bytes.to_vec()).expect("utf8 body"), headers)
}

/// Poll `f` (5ms apart) until true or the timeout - the deterministic
/// synchronization point for the queue-fill scenario.
async fn wait_until(f: impl Fn() -> bool, timeout_secs: u64) {
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    while !f() {
        assert!(Instant::now() < deadline, "condition not met within {timeout_secs}s");
        ntex::time::sleep(ntex::time::Millis(5)).await;
    }
}

/// The protocol 422 envelope: `{"error":{message,type,code,param,details[]}}`.
fn assert_422(status: &StatusCode, body: &str, detail_hint: &str) {
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

// ---------------------------------------------------------------------------
// Validation-matrix cases (weight-free; each asserts the protocol status +
// the error envelope defined in docs/jev-server/protocol-reference.md)
// ---------------------------------------------------------------------------

/// A valid mixed request body (one choice, one score, one noul) for `MODEL` -
/// the same refund example the fixtures use.
fn mixed_body() -> String {
    json!({
        "model": MODEL,
        "state": "We were billed twice for March. Please refund the duplicate.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": { "billing": "invoices, payments, refunds", "technical": null }
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]
            },
            "refund": {
                "type": "noul",
                "instructions": "Does the customer explicitly request a refund?",
                "criteria": { "true": "explicit refund ask", "false": "no refund ask" }
            }
        }
    })
    .to_string()
}

/// A choice-question body with exactly `n` criteria keys under question id
/// `route` (plain string building - no format-string brace escaping).
fn choice_body_with(n: usize) -> String {
    let keys: Vec<String> = (0..n).map(|i| format!("\"c{}\": null", i)).collect();
    let mut s = String::from("{\"model\":\"");
    s.push_str(MODEL);
    s.push_str(
        "\",\"state\":\"x\",\"questions\":{\"route\":{\"type\":\"choice\",\"instructions\":\"Which one?\",\"criteria\":{",
    );
    s.push_str(&keys.join(","));
    s.push_str("}}}}");
    s
}

/// A score-question body with exactly `n` criteria levels under `urgency`.
fn score_body_with(n: usize) -> String {
    let items: Vec<String> = (0..n).map(|i| format!("\"level {i}\"")).collect();
    let mut s = String::from("{\"model\":\"");
    s.push_str(MODEL);
    s.push_str(
        "\",\"state\":\"x\",\"questions\":{\"urgency\":{\"type\":\"score\",\"instructions\":\"How urgent?\",\"criteria\":[",
    );
    s.push_str(&items.join(","));
    s.push_str("]}}}");
    s
}

/// `n` minimal noul questions under `state: "x"`.
fn noul_body_with(n: usize) -> String {
    let qs: Vec<String> = (0..n)
        .map(|i| format!("\"q{}\":{{\"type\":\"noul\",\"instructions\":\"x?\"}}", i))
        .collect();
    let mut s = String::from("{\"model\":\"");
    s.push_str(MODEL);
    s.push_str("\",\"state\":\"x\",\"questions\":{");
    s.push_str(&qs.join(","));
    s.push_str("}}");
    s
}

async fn case_context_xor(app: &TestServer) {
    // both state and messages supplied
    let both = json!({
        "model": MODEL,
        "state": "x",
        "messages": [{ "role": "user", "content": "x" }],
        "questions": { "q1": { "type": "noul", "instructions": "x?" } }
    })
    .to_string();
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&both), Some("application/json")).await;
    assert_422(&s, &b, "Provide exactly one of state or messages");

    // neither supplied
    let neither = json!({
        "model": MODEL,
        "questions": { "q1": { "type": "noul", "instructions": "x?" } }
    })
    .to_string();
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&neither), Some("application/json")).await;
    assert_422(&s, &b, "Provide exactly one of state or messages");

    // state: null + messages: null is "neither supplied" too
    let both_null = json!({
        "model": MODEL,
        "state": null,
        "messages": null,
        "questions": { "q1": { "type": "noul", "instructions": "x?" } }
    })
    .to_string();
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&both_null), Some("application/json")).await;
    assert_422(&s, &b, "Provide exactly one of state or messages");

    // state: null + messages: a null state is *not* supplied (protocol
    // reference: "null is not supplied"), so messages alone satisfies the XOR
    let null_state = json!({
        "model": MODEL,
        "state": null,
        "messages": [{ "role": "user", "content": "x" }],
        "questions": { "q1": { "type": "noul", "instructions": "x?" } }
    })
    .to_string();
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&null_state), Some("application/json")).await;
    assert_eq!(s, StatusCode::OK, "null state + messages must be accepted: {b}");
}

async fn case_unknown_top_level(app: &TestServer, mock: &MockAnswerer) {
    let mut v: Value = serde_json::from_str(&mixed_body()).expect("mixed body is JSON");
    v["stream"] = json!(true);
    v["temperature"] = json!(0.7);
    v["max_tokens"] = json!(128);
    let body = v.to_string();
    let before = mock.forwards();
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&body), Some("application/json")).await;
    assert_eq!(s, StatusCode::OK, "unknown top-level fields are ignored, not rejected: {b}");
    let parsed: Value = serde_json::from_str(&b).expect("200 body is JSON");
    for qid in ["route", "urgency", "refund"] {
        assert!(parsed["answers"].get(qid).is_some(), "all questions answered: {b}");
    }
    assert_eq!(mock.forwards(), before + 1, "the request must have forwarded");
}

async fn case_unknown_question_field(app: &TestServer) {
    let body = json!({
        "model": MODEL,
        "state": "x",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team?",
                "context": "stray question field",
                "criteria": { "a": null, "b": null }
            }
        }
    })
    .to_string();
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&body), Some("application/json")).await;
    assert_422(&s, &b, "unknown question field");
    let parsed: Value = serde_json::from_str(&b).expect("422 body is JSON");
    assert_eq!(
        parsed["error"]["param"], "questions.route.context",
        "the envelope param must name the offending field: {b}"
    );
}

async fn case_type_discriminator(app: &TestServer) {
    let body = json!({
        "model": MODEL,
        "state": "x",
        "questions": { "q1": { "type": "regex", "instructions": "x?", "criteria": { "a": null, "b": null } } }
    })
    .to_string();
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&body), Some("application/json")).await;
    assert_422(&s, &b, "type must be 'choice', 'score', or 'noul'");
}

async fn case_criteria_cardinality(app: &TestServer) {
    // choice: 1 -> 422, 2 -> 200, 50 -> 200, 51 -> 422
    let (s1, b1) = hit(app, "POST", "/v1/classifier", Some(&choice_body_with(1)), Some("application/json")).await;
    assert_422(&s1, &b1, "choice criteria requires 2-50 candidates");
    let (s2, b2) = hit(app, "POST", "/v1/classifier", Some(&choice_body_with(2)), Some("application/json")).await;
    assert_eq!(s2, StatusCode::OK, "2 candidates is the choice minimum: {b2}");
    let (s50, b50) = hit(app, "POST", "/v1/classifier", Some(&choice_body_with(50)), Some("application/json")).await;
    assert_eq!(s50, StatusCode::OK, "50 candidates is the choice maximum: {b50}");
    let (s51, b51) = hit(app, "POST", "/v1/classifier", Some(&choice_body_with(51)), Some("application/json")).await;
    assert_422(&s51, &b51, "choice criteria allows at most 50 candidates");

    // score: empty -> 422, 1 -> 422, 2 -> 200, 50 -> 200, 51 -> 422
    let (s0, b0) = hit(app, "POST", "/v1/classifier", Some(&score_body_with(0)), Some("application/json")).await;
    assert_422(&s0, &b0, "score criteria requires 2-50 levels");
    let (s1s, b1s) = hit(app, "POST", "/v1/classifier", Some(&score_body_with(1)), Some("application/json")).await;
    assert_422(&s1s, &b1s, "score criteria requires 2-50 levels");
    let (s2s, b2s) = hit(app, "POST", "/v1/classifier", Some(&score_body_with(2)), Some("application/json")).await;
    assert_eq!(s2s, StatusCode::OK, "2 levels is the score minimum: {b2s}");
    let (s50s, b50s) = hit(app, "POST", "/v1/classifier", Some(&score_body_with(50)), Some("application/json")).await;
    assert_eq!(s50s, StatusCode::OK, "50 levels is the score maximum: {b50s}");
    let (s51s, b51s) = hit(app, "POST", "/v1/classifier", Some(&score_body_with(51)), Some("application/json")).await;
    assert_422(&s51s, &b51s, "score criteria allows at most 50 levels");
}

async fn case_noul_key_restriction(app: &TestServer) {
    // a 'maybe' key -> 422
    let maybe = json!({
        "model": MODEL,
        "state": "x",
        "questions": { "q1": { "type": "noul", "instructions": "x?", "criteria": { "maybe": "50%" } } }
    })
    .to_string();
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&maybe), Some("application/json")).await;
    assert_422(&s, &b, "noul criteria accepts only the 'true' and 'false' keys");

    // only 'true' -> 200 (either or both may be provided)
    let true_only = json!({
        "model": MODEL,
        "state": "x",
        "questions": { "q1": { "type": "noul", "instructions": "x?", "criteria": { "true": "affirmative" } } }
    })
    .to_string();
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&true_only), Some("application/json")).await;
    assert_eq!(s, StatusCode::OK, "a noul with only a 'true' key is valid: {b}");
}

async fn case_empty_question_id(app: &TestServer) {
    let body = json!({
        "model": MODEL,
        "state": "x",
        "questions": { "": { "type": "noul", "instructions": "x?" } }
    })
    .to_string();
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&body), Some("application/json")).await;
    assert_422(&s, &b, "question ids must not be empty");
}

async fn case_too_many_questions(app: &TestServer) {
    let body = noul_body_with(257);
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&body), Some("application/json")).await;
    assert_422(&s, &b, "at most 256 questions are allowed");
}

async fn case_state_type_checks(app: &TestServer) {
    // bare number / bare boolean -> 422
    for bad in [json!(42), json!(true)] {
        let body = json!({
            "model": MODEL,
            "state": bad,
            "questions": { "q1": { "type": "noul", "instructions": "x?" } }
        })
        .to_string();
        let (s, b) = hit(app, "POST", "/v1/classifier", Some(&body), Some("application/json")).await;
        assert_422(&s, &b, "state must be a string, JSON object, or JSON array");
    }
    // empty string / empty object / empty array all count as supplied state -> 200
    for good in [json!(""), json!({}), json!([])] {
        let body = json!({
            "model": MODEL,
            "state": good,
            "questions": { "q1": { "type": "noul", "instructions": "x?" } }
        })
        .to_string();
        let (s, b) = hit(app, "POST", "/v1/classifier", Some(&body), Some("application/json")).await;
        assert_eq!(s, StatusCode::OK, "empty state values are valid: {b}");
    }
}

async fn case_malformed_json(app: &TestServer) {
    let (s, b) = hit(app, "POST", "/v1/classifier", Some("{ this is not json"), Some("application/json")).await;
    assert_422(&s, &b, "invalid JSON");
}

async fn case_oversized_body(app: &TestServer, mock: &MockAnswerer) {
    // The 1 MiB limit exceeded by one byte of payload.
    let big = json!({ "model": MODEL, "state": "x".repeat(1048576), "questions": { "q1": { "type": "noul", "instructions": "x?" } } })
        .to_string();
    let before = mock.forwards();
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&big), Some("application/json")).await;
    assert_422(&s, &b, "1 MiB");
    assert_eq!(mock.forwards(), before, "an oversized body must never reach inference");
}

async fn case_wrong_content_type(app: &TestServer, mock: &MockAnswerer) {
    let body = mixed_body();
    let before = mock.forwards();
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&body), Some("text/plain")).await;
    assert_422(&s, &b, "application/json");
    assert_eq!(mock.forwards(), before, "a wrong content type must never reach inference");
}

/// Context overflow is rejected at admission - 422 naming the question, with
/// the model never called (the counting mock proves the no-forward ordering).
async fn case_overflow_admission() {
    let mock = MockAnswerer::with_admit_error(AdmissionError::ExceedsTokenCap {
        question: "route".to_string(),
        cap: 1024,
    });
    let state = ServerState::new(mock.clone(), MODEL, ServerConfig::default());
    let app = build_router(state).await;
    let (s, b) = hit(&app, "POST", "/v1/classifier", Some(&mixed_body()), Some("application/json")).await;
    assert_422(&s, &b, "exceeds 1024 input tokens");
    assert_eq!(mock.forwards(), 0, "admission must reject before any forward (no inference)");
}

async fn case_raw_logits(app: &TestServer) {
    let body = json!({
        "model": MODEL,
        "state": "x",
        "options": { "raw_logits": true },
        "questions": { "q1": { "type": "noul", "instructions": "x?" } }
    })
    .to_string();
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&body), Some("application/json")).await;
    assert_422(&s, &b, "raw_logits diagnostics are not supported by the laya backend");
}

async fn case_reserved_fields(app: &TestServer) {
    // nonempty reserved containers -> 422
    let tools = json!({
        "model": MODEL,
        "state": "x",
        "tools": [{ "type": "function" }],
        "questions": { "q1": { "type": "noul", "instructions": "x?" } }
    })
    .to_string();
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&tools), Some("application/json")).await;
    assert_422(&s, &b, "reserved and must be omitted or empty");

    // empty reserved containers -> accepted, no effect
    let empty = json!({
        "model": MODEL,
        "state": "x",
        "tools": [],
        "mm_processor_kwargs": {},
        "media_io_kwargs": {},
        "questions": { "q1": { "type": "noul", "instructions": "x?" } }
    })
    .to_string();
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&empty), Some("application/json")).await;
    assert_eq!(s, StatusCode::OK, "empty reserved containers are accepted: {b}");
}

async fn case_media_tool_rejection(app: &TestServer) {
    // image-part content -> 422 (text messages only)
    let image = json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": [{ "type": "image_url", "image_url": { "url": "https://example.com/a.png" } }] }],
        "questions": { "q1": { "type": "noul", "instructions": "x?" } }
    })
    .to_string();
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&image), Some("application/json")).await;
    assert_422(&s, &b, "content must be a string (the laya backend supports text messages only)");

    // tool role -> 422
    let tool_role = json!({
        "model": MODEL,
        "messages": [{ "role": "tool", "content": "x" }],
        "questions": { "q1": { "type": "noul", "instructions": "x?" } }
    })
    .to_string();
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&tool_role), Some("application/json")).await;
    assert_422(&s, &b, "text messages only");

    // extra message field (name) -> 422
    let extra = json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "x", "name": "bot" }],
        "questions": { "q1": { "type": "noul", "instructions": "x?" } }
    })
    .to_string();
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(&extra), Some("application/json")).await;
    assert_422(&s, &b, "text-only messages");
}

/// Chat history (`messages` instead of `state`) is answered over the wire.
async fn case_chat_history(app: &TestServer, mock: &MockAnswerer) {
    let before = mock.forwards();
    let body = include_str!("jev_conformance/chat_history.json");
    let (s, b) = hit(app, "POST", "/v1/classifier", Some(body), Some("application/json")).await;
    assert_eq!(s, StatusCode::OK, "chat history must be answered: {b}");
    let parsed: Value = serde_json::from_str(&b).expect("200 body is JSON");
    assert_eq!(parsed["answers"]["route"]["type"], "choice", "the message context was scored: {b}");
    assert_eq!(mock.forwards(), before + 1, "the messages path must reach inference");
}

/// More questions than `--max-request-branches` -> 422 (separate server config).
async fn case_branch_limit() {
    let mock = Arc::new(MockAnswerer::default());
    let state = ServerState::new(mock.clone(), MODEL, ServerConfig { max_request_branches: 2, ..Default::default() });
    let app = build_router(state).await;
    let body = noul_body_with(3);
    let (s, b) = hit(&app, "POST", "/v1/classifier", Some(&body), Some("application/json")).await;
    assert_422(&s, &b, "3 questions exceed the 2-branch request limit");
    assert_eq!(mock.forwards(), 0, "the branch limit must reject before any forward");
}

/// The 429 admission queue: `max_queued: 1` + a blocking mock forward fills
/// the queue; the next request gets `{"detail":"Scoring queue is full"}` +
/// `Retry-After: 1`, and never reaches the model.
async fn case_queue_full_429_retry_after() {
    let gate = Arc::new(ForwardGate::default());
    let mock = MockAnswerer::with_gate(gate.clone());
    let config = ServerConfig { max_queued: 1, ..Default::default() };
    let state = ServerState::new(mock.clone(), MODEL, config);
    let probe = state.clone();
    let app = Rc::new(build_router(state).await);
    let body = mixed_body();

    // A: the in-flight forward (holds the one in-flight permit + the model
    // lock; its mock forward parks on the gate).
    let body_a = body.clone();
    let app_a = Rc::clone(&app);
    let a_task = ntex::rt::spawn(async move {
        hit(&app_a, "POST", "/v1/classifier", Some(&body_a), Some("application/json")).await
    });
    wait_until(|| gate.started(), 10).await;

    // B: the single waiting slot (admitted, then blocks behind A's model lock).
    let body_b = body.clone();
    let app_b = Rc::clone(&app);
    let b_task = ntex::rt::spawn(async move {
        hit(&app_b, "POST", "/v1/classifier", Some(&body_b), Some("application/json")).await
    });
    wait_until(|| probe.admission_available_slots() == 0, 10).await;

    // C: the queue is full -> 429 + Retry-After: 1.
    let (status, body_c, headers) =
        hit_full(&app, "POST", "/v1/classifier", Some(&body), Some("application/json")).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "a full queue must 429: {body_c}");
    assert_eq!(body_c, "{\"detail\":\"Scoring queue is full\"}", "the 429 short form");
    assert_eq!(
        headers.get(header::RETRY_AFTER).and_then(|v| v.to_str().ok()),
        Some("1"),
        "Retry-After must be 1"
    );
    assert_eq!(mock.forwards(), 1, "only A may have forwarded so far");

    // Release A: A and B complete in turn; the server stays up.
    gate.release();
    let (sa, ba) = a_task.await.expect("A joins");
    assert_eq!(sa, StatusCode::OK, "A: {ba}");
    let (sb, bb) = b_task.await.expect("B joins");
    assert_eq!(sb, StatusCode::OK, "B: {bb}");
    assert_eq!(mock.forwards(), 2, "exactly A and B forwarded; the 429 request never reached the model");

    let (s3, b3) = hit(&app, "POST", "/v1/classifier", Some(&body), Some("application/json")).await;
    assert_eq!(s3, StatusCode::OK, "the server must keep serving after the 429: {b3}");
}

/// A forward error surfaces as 500 `{"detail":"internal error"}` and the
/// server keeps serving.
async fn case_forward_error_500_then_recovery() {
    let mock = Arc::new(MockAnswerer { fail_once: AtomicBool::new(true), ..Default::default() });
    let state = ServerState::new(mock.clone(), MODEL, ServerConfig::default());
    let app = build_router(state).await;
    let body = mixed_body();

    let (s1, b1) = hit(&app, "POST", "/v1/classifier", Some(&body), Some("application/json")).await;
    assert_eq!(s1, StatusCode::INTERNAL_SERVER_ERROR, "a forward error must 500: {b1}");
    assert_eq!(b1, "{\"detail\":\"internal error\"}", "the 500 short form");

    let (s2, b2) = hit(&app, "POST", "/v1/classifier", Some(&body), Some("application/json")).await;
    assert_eq!(s2, StatusCode::OK, "the server must stay up after the 500: {b2}");
    assert_eq!(mock.forwards(), 2, "exactly the two non-cancelled requests may forward");
}

async fn case_alias_byte_equivalent(app: &TestServer) {
    let body = mixed_body();
    let (s_main, b_main) = hit(app, "POST", "/v1/classifier", Some(&body), Some("application/json")).await;
    let (s_alias, b_alias) = hit(app, "POST", "/v1/systemone", Some(&body), Some("application/json")).await;
    assert_eq!(s_main, StatusCode::OK, "classifier must serve the request: {b_main}");
    assert_eq!(s_alias, StatusCode::OK, "the alias must serve the request: {b_alias}");
    assert_eq!(b_main, b_alias, "byte-equivalent bodies: main={b_main} alias={b_alias}");
    let parsed: Value = serde_json::from_str(&b_main).expect("200 body is JSON");
    assert_eq!(parsed["model"], MODEL);
    for qid in ["route", "urgency", "refund"] {
        assert!(parsed["answers"].get(qid).is_some(), "answers must name the question id {qid}: {b_main}");
    }
    assert_eq!(parsed["usage"]["output_tokens"], 0, "usage.output_tokens must be 0");
    // usage.input_tokens = the sum of the mock's per-question admit lengths (16 * 3)
    assert_eq!(parsed["usage"]["input_tokens"], 48, "usage.input_tokens must sum the admit lengths: {b_main}");
}

async fn case_health(app: &TestServer, mock: &MockAnswerer) {
    let before = mock.forwards();
    let (s, b) = hit(app, "GET", "/health", None, None).await;
    assert_eq!(s, StatusCode::OK, "health: {b}");
    let v: Value = serde_json::from_str(&b).expect("health body is JSON");
    assert_eq!(v["status"], "ready");
    assert_eq!(v["model"], MODEL);
    assert_eq!(mock.forwards(), before, "health must not trigger a forward pass");
}

async fn case_openapi_shape(app: &TestServer) {
    let (s, b) = hit(app, "GET", "/openapi.json", None, None).await;
    assert_eq!(s, StatusCode::OK, "openapi: {b}");
    let v: Value = serde_json::from_str(&b).expect("openapi.json must be valid JSON");
    assert!(v["openapi"].as_str().is_some_and(|s| s.starts_with("3.1")), "must be OpenAPI 3.1: {v}");
    for path in ["/v1/classifier", "/v1/systemone", "/health"] {
        assert!(v["paths"].get(path).is_some(), "paths must describe {path}: {v}");
    }
}

/// The full weight-free matrix, run against one fresh router + mock.
async fn run_validation_matrix() {
    let mock = Arc::new(MockAnswerer::default());
    let app = {
        let state = ServerState::new(mock.clone(), MODEL, ServerConfig::default());
        build_router(state).await
    };

    case_context_xor(&app).await;
    case_unknown_top_level(&app, &mock).await;
    case_unknown_question_field(&app).await;
    case_type_discriminator(&app).await;
    case_criteria_cardinality(&app).await;
    case_noul_key_restriction(&app).await;
    case_empty_question_id(&app).await;
    case_too_many_questions(&app).await;
    case_state_type_checks(&app).await;
    case_malformed_json(&app).await;
    case_oversized_body(&app, &mock).await;
    case_wrong_content_type(&app, &mock).await;
    case_raw_logits(&app).await;
    case_reserved_fields(&app).await;
    case_media_tool_rejection(&app).await;
    case_chat_history(&app, &mock).await;

    // cases with their own state/config (separate from the shared router)
    case_overflow_admission().await;
    case_branch_limit().await;
}

// ---------------------------------------------------------------------------
// Model-gated tier: a real checkpoint behind start_server (LAYA_TEST_MODEL)
// ---------------------------------------------------------------------------

fn model_dir() -> Option<String> {
    std::env::var("LAYA_TEST_MODEL").ok()
}

/// A live server: the real checkpoint behind `start_server` (port 0 = an
/// ephemeral port), plus the checkpoint's native limits (the overflow case
/// sizes its state against `max_len`).
struct LiveServer {
    handle: ServerHandle,
    limits: (usize, usize),
}

async fn start_live(model: &str) -> LiveServer {
    let model = model.to_string();
    let load_model = model.clone();
    let (limits, answerer): ((usize, usize), Arc<dyn Answerer>) =
        ntex::rt::spawn_blocking(move || {
            let agent = RLAgent::load(&load_model).unwrap_or_else(|e| panic!("loading {load_model}: {e}"));
            let limits = agent.checkpoint_limits();
            let answerer: Arc<dyn Answerer> = Arc::new(RealAnswerer::new(Arc::new(agent)));
            (limits, answerer)
        })
        .await
        .expect("checkpoint load task joins");
    let handle = start_server("127.0.0.1", 0, answerer, model.clone(), ServerConfig::default())
        .await
        .expect("start_server on an ephemeral port");
    LiveServer { handle, limits }
}

/// Minimal HTTP/1.1 round-trip over a bare `TcpStream` (the test deps carry no
/// HTTP client - the tests/serve_cli.rs pattern): returns `(status, body)`, or
/// `None` when the connection fails (server not up yet / already stopped).
fn http(host_port: &str, method: &str, path: &str, body: Option<&str>) -> Option<(u16, String)> {
    let mut stream = TcpStream::connect(host_port).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(120))).ok()?;
    let body = body.unwrap_or("");
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes()).ok()?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).ok()?;
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, body) = text.split_once("\r\n\r\n")?;
    let status = head.lines().next()?.split_whitespace().nth(1)?.parse().ok()?;
    Some((status, body.to_string()))
}

/// The blocking round-trip offloaded to the blocking pool (a bare `TcpStream`
/// read must not block a runtime worker while the serve task needs one).
async fn http_async(host_port: &str, method: &str, path: &str, body: Option<&str>) -> Option<(u16, String)> {
    let host_port = host_port.to_string();
    let method = method.to_string();
    let path = path.to_string();
    let body_owned = body.map(|b| b.to_string());
    ntex::rt::spawn_blocking(move || http(&host_port, &method, &path, body_owned.as_deref()))
        .await
        .expect("http task joins")
}

/// Poll `GET /health` until 200, or `None` on timeout.
async fn wait_for_health(host_port: &str, timeout: Duration) -> Option<String> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some((status, body)) = http_async(host_port, "GET", "/health", None).await {
            if status == 200 {
                return Some(body);
            }
        }
        if Instant::now() >= deadline {
            return None;
        }
        ntex::time::sleep(ntex::time::Millis(50)).await;
    }
}

/// Swap the fixture's model field for the loaded checkpoint's identity,
/// preserving document order (a `Value` round-trip would BTree-sort the
/// questions keys and break the request-order assertions).
fn with_model(raw: &str, model: &str) -> String {
    let model_json = serde_json::to_string(model).expect("string serializes");
    let swapped = raw.replace("\"model\": \"convaiinnovations/laya\"", &format!("\"model\": {model_json}"));
    assert_ne!(swapped, raw, "the fixture must carry the placeholder model field");
    swapped
}

/// The model-gated contract round-trips (skips when `LAYA_TEST_MODEL` unset):
/// mixed choice/score/noul shape, wrong-model 422, context-overflow 422,
/// chat-history 200, and determinism (identical request -> identical bytes).
async fn contract_round_trips_body() {
    let model = match model_dir() {
        Some(d) => d,
        None => return, // skipped: no checkpoint available (LAYA_TEST_MODEL unset)
    };
    let live = start_live(&model).await;
    let host_port = format!("127.0.0.1:{}", live.handle.port);
    wait_for_health(&host_port, Duration::from_secs(60))
        .await
        .expect("the live server must become ready");

    // 1. The mixed choice/score/noul fixture (README's refund example) -> 200,
    //    protocol-shaped answers.
    let mixed_str = with_model(include_str!("jev_conformance/mixed_fixture.json"), &model);
    let (status, body) = http_async(&host_port, "POST", "/v1/classifier", Some(&mixed_str)).await.expect("reachable");
    assert_eq!(status, 200, "mixed round-trip must 200: {body}");
    let v: Value = serde_json::from_str(&body).expect("200 body must be JSON");
    assert_eq!(v["model"], model);
    for qid in ["route", "urgency", "refund"] {
        assert!(v["answers"].get(qid).is_some(), "answers must name {qid}: {body}");
    }
    // answers follow the fixture's document order (route, urgency, refund)
    let i_route = body.find("\"route\"").expect("route answer present");
    let i_urgency = body.find("\"urgency\"").expect("urgency answer present");
    let i_refund = body.find("\"refund\"").expect("refund answer present");
    assert!(i_route < i_urgency && i_urgency < i_refund, "answers must follow request order: {body}");

    let route = &v["answers"]["route"];
    assert_eq!(route["type"], "choice");
    assert!(
        route.get("choice").is_some() && route.get("confidence").is_some() && route.get("probabilities").is_some(),
        "choice shape: {route}"
    );
    assert!(route.get("act_probability").is_none(), "no act_probability on a choice answer: {route}");
    let urgency = &v["answers"]["urgency"];
    assert_eq!(urgency["type"], "score");
    assert!(
        urgency.get("score").is_some() && urgency.get("confidence").is_some() && urgency.get("legend").is_some(),
        "score shape: {urgency}"
    );
    let refund = &v["answers"]["refund"];
    assert_eq!(refund["type"], "noul");
    assert!(refund.get("noul").is_some(), "noul value present: {refund}");
    assert!(refund.get("confidence").is_none(), "noul carries no confidence: {refund}");
    assert_eq!(refund.as_object().map(|o| o.len()), Some(2), "noul is exactly type+noul, no more: {refund}");

    assert_eq!(v["usage"]["output_tokens"], 0, "usage.output_tokens must be 0: {body}");
    assert!(v["usage"]["input_tokens"].as_u64().is_some_and(|n| n > 0), "usage.input_tokens must be > 0: {body}");

    // 2. Wrong model -> 422 naming the loaded model.
    let (status, body) =
        http_async(&host_port, "POST", "/v1/classifier", Some(include_str!("jev_conformance/wrong_model.json"))).await.expect("reachable");
    assert_eq!(status, 422, "the wrong-model request must 422: {body}");
    let e: Value = serde_json::from_str(&body).expect("422 body is JSON");
    assert_eq!(e["error"]["type"], "invalid_request_error");
    assert!(
        e["error"]["message"].as_str().is_some_and(|m| m.contains(&format!("Loaded model is '{model}'"))),
        "the 422 must name the loaded model: {body}"
    );

    // 3. Context overflow: a state far beyond the checkpoint's native
    // max_len -> 422 at admission, before any forward (the weight-free tier
    // pins the no-forward ordering with a counting mock; here the real
    // tokenizer's admission arithmetic must agree on a genuinely over-long
    // state: "x " repeated 2*max_len times is at least 2*max_len tokens).
    let (max_len, _) = live.limits;
    let long_state = "x ".repeat(max_len * 2);
    let long_body = json!({
        "model": model,
        "state": long_state,
        "questions": {
            "route": { "type": "choice", "instructions": "Which team?", "criteria": { "billing": null, "technical": null } }
        }
    })
    .to_string();
    let (status, body) = http_async(&host_port, "POST", "/v1/classifier", Some(&long_body)).await.expect("reachable");
    assert_eq!(status, 422, "the over-long state must 422 without inference: {body}");
    let e: Value = serde_json::from_str(&body).expect("422 body is JSON");
    assert!(
        e["error"]["message"].as_str().is_some_and(|m| m.contains("exceeds") && m.contains("route")),
        "the 422 must name the overflowing question and the cap: {body}"
    );

    // 4. Chat history (messages instead of state) -> 200, answered against the
    //    serialized message list.
    let chat_str = with_model(include_str!("jev_conformance/chat_history.json"), &model);
    let (status, body) = http_async(&host_port, "POST", "/v1/classifier", Some(&chat_str)).await.expect("reachable");
    assert_eq!(status, 200, "the chat-history request must be answered: {body}");
    let v: Value = serde_json::from_str(&body).expect("200 body is JSON");
    assert_eq!(v["answers"]["route"]["type"], "choice", "the messages context was scored: {body}");

    // 5. Determinism: two identical requests -> byte-identical responses
    //    (laya is non-autoregressive; nothing is sampled).
    let (_, b1) = http_async(&host_port, "POST", "/v1/classifier", Some(&mixed_str)).await.expect("reachable");
    let (status, b2) = http_async(&host_port, "POST", "/v1/classifier", Some(&mixed_str)).await.expect("reachable");
    assert_eq!(status, 200, "the repeated request must 200: {b2}");
    assert_eq!(b1, b2, "identical requests must return byte-identical responses");

    live.handle.stop().await;
}

/// Numerical sanity + determinism on a real checkpoint (skips when
/// `LAYA_TEST_MODEL` unset): choice probabilities sum ~ 1 with confidence =
/// max, score in [0, N-1], noul in [0, 1], repeated request identical.
async fn numerical_sanity_body() {
    let model = match model_dir() {
        Some(d) => d,
        None => return, // skipped: no checkpoint available (LAYA_TEST_MODEL unset)
    };
    let live = start_live(&model).await;
    let host_port = format!("127.0.0.1:{}", live.handle.port);
    wait_for_health(&host_port, Duration::from_secs(60))
        .await
        .expect("the live server must become ready");

    let body = with_model(include_str!("jev_conformance/mixed_fixture.json"), &model);
    let (status, resp) = http_async(&host_port, "POST", "/v1/classifier", Some(&body)).await.expect("reachable");
    assert_eq!(status, 200, "the sanity request must 200: {resp}");
    let v: Value = serde_json::from_str(&resp).expect("200 body is JSON");

    // choice: probabilities sum within 0.01 of 1; confidence = max probability;
    // the winning candidate carries that max.
    let route = &v["answers"]["route"];
    let probs: Vec<f64> = route["probabilities"]
        .as_object()
        .expect("probabilities must be an object")
        .values()
        .map(|p| p.as_f64().expect("probability must be numeric"))
        .collect();
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() <= 0.01, "choice probabilities must sum to 1 (got {sum})");
    let max = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let confidence = route["confidence"].as_f64().expect("confidence must be numeric");
    assert!((confidence - max).abs() <= 1e-9, "confidence {confidence} must equal the max probability {max}");
    let choice = route["choice"].as_str().expect("choice id");
    assert_eq!(route["probabilities"][choice].as_f64(), Some(max), "the winning candidate carries the max probability");

    // score: within [0, N-1] (N = 3 levels in the fixture).
    let score = v["answers"]["urgency"]["score"].as_f64().expect("score must be numeric");
    assert!((0.0..=2.0).contains(&score), "score {score} must lie in [0, N-1] for a 3-level rubric");

    // noul: within [0, 1].
    let noul = v["answers"]["refund"]["noul"].as_f64().expect("noul must be numeric");
    assert!((0.0..=1.0).contains(&noul), "noul {noul} must lie in [0, 1]");

    // determinism: a repeated identical request returns identical answers.
    let (status, resp2) = http_async(&host_port, "POST", "/v1/classifier", Some(&body)).await.expect("reachable");
    assert_eq!(status, 200, "the repeated request must 200: {resp2}");
    let v2: Value = serde_json::from_str(&resp2).expect("200 body is JSON");
    assert_eq!(v["answers"], v2["answers"], "identical request twice must yield identical answers");

    live.handle.stop().await;
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

/// Scenario: The weight-free tier passes with no checkpoint and no GPU
#[ntex::test]
async fn the_weight_free_tier_passes_with_no_checkpoint_and_no_gpu() {
    // @step Given the server router is built with a mock answerer
    // (each case below builds its own `build_router(ServerState::new(mock, ...))`
    //  from the mock `Answerer` - no checkpoint is loaded anywhere in this test)
    // @step When the suite runs the validation matrix, alias, health, openapi shape, 429-queue (blocking mock), and 500-recovery scenarios
    run_validation_matrix().await;
    let mock = Arc::new(MockAnswerer::default());
    let app = {
        let state = ServerState::new(mock.clone(), MODEL, ServerConfig::default());
        build_router(state).await
    };
    case_alias_byte_equivalent(&app).await;
    case_health(&app, &mock).await;
    case_openapi_shape(&app).await;
    case_queue_full_429_retry_after().await;
    case_forward_error_500_then_recovery().await;

    // @step Then all of them pass on a plain `cargo test` with no checkpoint, GPU, or LAYA_TEST_MODEL set
    // (reaching this line without a panic is the pass; the tier needs no
    //  checkpoint and no GPU - the mock is the entire answerer)
}

/// Scenario: Model-gated scenarios skip cleanly without a checkpoint
#[ntex::test]
async fn model_gated_scenarios_skip_cleanly_without_a_checkpoint() {
    // @step Given LAYA_TEST_MODEL is unset
    // (with a checkpoint available the gated tier runs instead of skipping -
    //  the suite passes either way, so the scenario is vacuous then)
    if model_dir().is_some() {
        return;
    }

    // @step When the suite runs the real round-trip scenarios (mixed choice/score/noul, determinism, numerical sanity)
    // (drive each gated body through its LAYA_TEST_MODEL gate: with no
    //  checkpoint directory they must take the skip path - return, not fail)
    contract_round_trips_body().await;
    numerical_sanity_body().await;

    // @step Then they are skipped (not failed) and the overall suite still passes
    // (reaching this line without a panic is the skip; the overall pass is what
    //  a plain `cargo test` reports)
}

/// Scenario: Model-gated contract round-trips assert the protocol shape
#[ntex::test]
async fn model_gated_contract_round_trips_assert_the_protocol_shape() {
    // @step Given LAYA_TEST_MODEL points at a real laya checkpoint
    // @step When the suite POSTs the mixed choice/score/noul fixture, a wrong-model request, and an over-long state
    // @step Then the mixed request returns protocol-shaped answers (no act_probability, noul without confidence, usage.output_tokens 0), the wrong-model request 422s, the over-long state 422s without invoking the model, and two identical requests return byte-identical answers
    // (all three steps are asserted inside contract_round_trips_body; the
    //  LAYA_TEST_MODEL gate there skips the whole scenario when unset)
    contract_round_trips_body().await;
}

/// Scenario: The suite pins the validation matrix
#[ntex::test]
async fn the_suite_pins_the_validation_matrix() {
    // @step Given the weight-free router with a mock answerer
    let mock = Arc::new(MockAnswerer::default());
    let app = {
        let state = ServerState::new(mock.clone(), MODEL, ServerConfig::default());
        build_router(state).await
    };

    // @step When the suite exercises context XOR, unknown top-level tolerance, unknown question fields, the type discriminator, criteria cardinality (1/2/50/51), noul key restriction, empty question id, state type checks, malformed JSON, oversized body, wrong content-type, a full queue (429 + Retry-After), and a forward error (500 then recovery)
    case_context_xor(&app).await;
    case_unknown_top_level(&app, &mock).await;
    case_unknown_question_field(&app).await;
    case_type_discriminator(&app).await;
    case_criteria_cardinality(&app).await;
    case_noul_key_restriction(&app).await;
    case_empty_question_id(&app).await;
    case_state_type_checks(&app).await;
    case_malformed_json(&app).await;
    case_oversized_body(&app, &mock).await;
    case_wrong_content_type(&app, &mock).await;
    case_queue_full_429_retry_after().await;
    case_forward_error_500_then_recovery().await;

    // @step Then each case returns the protocol status and error envelope defined in the protocol reference, and the server stays up after the 500
    // (each case asserts its own protocol status + envelope above; the
    //  recovery case ends with a successful 200 on its router - the server
    //  stayed up after the 500. The extra matrix cases (overflow, branch
    //  limit, raw_logits, reserved fields, media rejection, chat history,
    //  256-question cap) are pinned by the weight-free tier scenario)
}

/// Scenario: Numerical sanity and determinism hold on a real checkpoint
#[ntex::test]
async fn numerical_sanity_and_determinism_hold_on_a_real_checkpoint() {
    // @step Given LAYA_TEST_MODEL points at a real laya checkpoint
    // @step When the suite answers a choice, a score, and a noul question
    // @step Then choice probabilities sum within 0.01 of 1 with confidence equal to the max probability, the score lies in [0, N-1], the noul lies in [0, 1], and a repeated identical request returns identical answers
    // (all three steps are asserted inside numerical_sanity_body; the
    //  LAYA_TEST_MODEL gate there skips the whole scenario when unset)
    numerical_sanity_body().await;
}
