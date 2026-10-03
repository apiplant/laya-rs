/**
 * Feature: spec/features/laya-serve-cli-subcommand-checkpoint-load-server-lifecycle-graceful-shutdown.feature
 *
 * This test file validates the acceptance criteria defined in the feature file.
 * Scenarios map directly to Gherkin scenarios.
 *
 * Two tiers (repo convention):
 * - Weight-free: parse-time flag rejection and clean load failure via the real
 *   `laya` binary (CARGO_BIN_EXE_laya, or LAYA_TEST_BINARY), the
 *   `--max-model-len` clamp at the `ServerState` seam (mock `Answerer` — no
 *   checkpoint, no GPU), and the `start_server` lifecycle (ephemeral port,
 *   /health round-trip, stop()) behind the `Answerer` seam.
 * - Model-gated (LAYA_TEST_MODEL, a checkpoint directory): the real-load
 *   lifecycle — load progress + banner + /health, port conflict, wrong-model
 *   422 identity, SIGTERM graceful exit 0. Skipped when unset.
 */

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use laya::schema::{QType, Question};
use laya::server::server::start_server;
use laya::server::state::{AdmissionError, Answerer, ServerConfig, ServerState};
use laya::Answer;
use serde_json::{json, Value};

fn laya_bin() -> String {
    std::env::var("LAYA_TEST_BINARY")
        .unwrap_or_else(|_| env!("CARGO_BIN_EXE_laya").to_string())
}

fn model_dir() -> Option<String> {
    std::env::var("LAYA_TEST_MODEL").ok()
}

/// A mock [`Answerer`] standing in for a loaded checkpoint: it reports fixed
/// native limits, records the effective cap `admit` is called with (the
/// clamp assertion), and its forward optionally sleeps (the "long in-flight
/// forward" the graceful-shutdown scenario needs).
struct MockAnswerer {
    limits: (usize, usize),
    forward_delay: Duration,
    admit_cap: std::sync::Mutex<Option<usize>>,
    forward_completed: AtomicBool,
}

impl MockAnswerer {
    fn new(limits: (usize, usize), forward_delay_ms: u64) -> Self {
        Self {
            limits,
            forward_delay: Duration::from_millis(forward_delay_ms),
            admit_cap: std::sync::Mutex::new(None),
            forward_completed: AtomicBool::new(false),
        }
    }
}

impl Answerer for MockAnswerer {
    fn checkpoint_limits(&self) -> (usize, usize) {
        self.limits
    }

    fn admit(
        &self,
        _state: &Value,
        questions: &[(String, Question)],
        effective_cap: usize,
    ) -> Result<Vec<usize>, AdmissionError> {
        *self.admit_cap.lock().expect("admit_cap poisoned") = Some(effective_cap);
        Ok(vec![5; questions.len()])
    }

    fn system_one(&self, _state: &Value, questions: &[(String, Question)]) -> anyhow::Result<Vec<(String, Answer)>> {
        if !self.forward_delay.is_zero() {
            std::thread::sleep(self.forward_delay);
        }
        self.forward_completed.store(true, Ordering::SeqCst);
        Ok(questions
            .iter()
            .map(|(qid, _)| {
                (
                    qid.clone(),
                    Answer::Choice {
                        choice: "a".into(),
                        probabilities: vec![("a".into(), 0.9), ("b".into(), 0.1)],
                        confidence: 0.9,
                        act_probability: 0.0,
                    },
                )
            })
            .collect())
    }
}

/// One minimal valid choice question (protocol-shape filler for the seam).
fn one_choice_question() -> Question {
    Question {
        qtype: QType::Choice,
        instructions: "Which team should handle this?".to_string(),
        choice_criteria: vec![("a".to_string(), None), ("b".to_string(), None)],
        score_criteria: vec![],
        noul_true: None,
        noul_false: None,
    }
}

/// One minimal valid classifier request body for `model`.
fn classifier_body(model: &str) -> String {
    json!({
        "model": model,
        "state": "We were billed twice for March.",
        "questions": {
            "q": { "type": "choice", "instructions": "Which team?", "criteria": { "a": null, "b": null } }
        }
    })
    .to_string()
}

/// Minimal HTTP/1.1 round-trip over a bare `TcpStream` (the test deps carry no
/// HTTP client): returns `(status, body)`, or `None` when the connection
/// fails (listener not up yet / already stopped).
fn http(host_port: &str, method: &str, path: &str, body: Option<&str>) -> Option<(u16, String)> {
    let mut stream = TcpStream::connect(host_port).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(10))).ok()?;
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

/// Poll `GET /health` until 200 (the server only listens after the
/// checkpoint is loaded, so this also waits out the load), or `None` on
/// timeout.
fn wait_for_health(host_port: &str, timeout: Duration) -> Option<String> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some((status, body)) = http(host_port, "GET", "/health", None) {
            if status == 200 {
                return Some(body);
            }
        }
        if Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// A free port on 127.0.0.1 (bind-to-0 trick).
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").expect("bind ephemeral").local_addr().expect("local addr").port()
}

/// Spawn `laya serve` against the LAYA_TEST_MODEL checkpoint on a free port
/// (with the models root pointed at a nonexistent dir so nothing can
/// download), and wait for /health.
fn spawn_serve(model: &str, port: u16) -> std::process::Child {
    let child = Command::new(laya_bin())
        .args(["serve", "--model", model, "--port", &port.to_string()])
        .env("LAYA_MODELS_ROOT", "/nonexistent")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn laya serve");
    let health = wait_for_health(&format!("127.0.0.1:{port}"), Duration::from_secs(180));
    match health {
        Some(_) => child,
        None => {
            let out = child.wait_with_output().expect("wait for serve child");
            panic!("server never became ready; stderr: {}", String::from_utf8_lossy(&out.stderr));
        }
    }
}

/// Scenario: Serve loads the checkpoint and serves on the requested port
#[ntex::test]
async fn serve_loads_the_checkpoint_and_serves_on_the_requested_port() {
    // @step Given a checkpoint at /path/to/laya-typed-decisions
    // (weight-free stand-in for the checkpoint: the `Answerer` seam reporting
    //  the checkpoint's limits; the real load is the model-gated half below)
    let supplied_model = "/path/to/laya-typed-decisions".to_string();
    let answerer: Arc<dyn Answerer> = Arc::new(MockAnswerer::new((1024, 128), 0));

    // @step When the user runs `laya serve --model /path/to/laya-typed-decisions --port 8000`
    // (seam: `start_server` with port 0 = an ephemeral stand-in for the
    //  requested port; the CLI half below serves on a real requested port)
    let handle = start_server("127.0.0.1", 0, answerer, supplied_model.clone(), ServerConfig::default())
        .await
        .expect("start_server on an ephemeral port");
    assert!(handle.port != 0, "the handle must report the actually-bound port");

    // @step Then the checkpoint loads with stderr progress, the server prints `laya serving '/path/to/laya-typed-decisions' on http://127.0.0.1:8000`, and /health reports that model name
    // (seam half: /health reports the supplied model name; the load progress
    //  line and the banner are asserted in the model-gated half)
    // (on the blocking pool: the ntex runtime thread must stay free to drive the server)
    let host_port = format!("127.0.0.1:{}", handle.port);
    let (status, body) = ntex::rt::spawn_blocking(move || http(&host_port, "GET", "/health", None))
        .await
        .expect("client task must complete")
        .expect("server must be reachable after start_server");
    assert_eq!(status, 200, "health must be 200");
    assert!(body.contains("\"status\":\"ready\""), "health body: {body}");
    assert!(
        body.contains(&supplied_model),
        "health must report the supplied model name: {body}"
    );
    handle.stop().await;

    // Model-gated half: the real binary with the real load.
    let model = match model_dir() {
        Some(dir) => dir,
        None => return, // skipped: no checkpoint available (LAYA_TEST_MODEL unset)
    };
    let port = free_port();
    let mut child = spawn_serve(&model, port);
    let health_body = http(&format!("127.0.0.1:{port}"), "GET", "/health", None)
        .expect("reachable after wait_for_health")
        .1;
    assert!(
        health_body.contains(&model),
        "health must report the --model path: {health_body}"
    );

    // (continued from the Then step above: load progress + banner on stderr)
    let _ = child.kill();
    let out = child.wait_with_output().expect("wait");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("loading"), "stderr must report checkpoint load progress: {stderr}");
    assert!(stderr.contains(&model), "load progress must name the checkpoint: {stderr}");
    assert!(stderr.contains("laya serving"), "the startup banner must be printed: {stderr}");
    assert!(
        stderr.contains(&format!("http://127.0.0.1:{port}")),
        "the banner must name the listen address: {stderr}"
    );
}

/// Scenario: Port conflicts and load failures fail cleanly before serving
#[test]
fn port_conflicts_and_load_failures_fail_cleanly_before_serving() {
    // @step And a checkpoint that cannot be loaded (missing files) fails the same way before any HTTP is served
    // (weight-free half first — it needs no checkpoint at all: the CLI loads
    //  before binding, so a missing dir fails at load, never at bind)
    let missing = "/definitely/not/a/real/checkpoint/dir";
    let out = Command::new(laya_bin())
        .args(["serve", "--model", missing, "--port", "0"])
        .output()
        .expect("run laya serve");
    assert!(!out.status.success(), "a missing checkpoint must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(missing),
        "the load error must name the checkpoint (readable): {stderr}"
    );
    assert!(!stderr.contains("laya serving"), "nothing may be served after a load failure: {stderr}");

    let model = match model_dir() {
        Some(dir) => dir,
        None => return, // skipped: no checkpoint available (LAYA_TEST_MODEL unset)
    };

    // @step Given another process already owns port 8000
    // (the 8000 stand-in: a test-owned listener on an ephemeral port)
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind the conflict port");
    let port = listener.local_addr().expect("local addr").port();

    // @step When the user runs `laya serve --port 8000`
    let out = Command::new(laya_bin())
        .args(["serve", "--model", &model, "--port", &port.to_string()])
        .env("LAYA_MODELS_ROOT", "/nonexistent")
        .output()
        .expect("run laya serve");

    // @step Then the command fails with a readable bind error, no HTTP is served, and the process exits non-zero
    assert!(
        !out.status.success(),
        "a port conflict must fail; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.to_lowercase().contains("bind"),
        "the error must name the bind failure: {stderr}"
    );
    assert!(!stderr.contains("laya serving"), "no HTTP may be served on a conflicting port: {stderr}");
}

/// Scenario: The effective sequence cap clamps to the checkpoint's native max_len
#[ntex::test]
async fn the_effective_sequence_cap_clamps_to_the_checkpoints_native_max_len() {
    // @step Given a checkpoint whose native max_len is 1024
    let mock = Arc::new(MockAnswerer::new((1024, 128), 0));
    let answerer: Arc<dyn Answerer> = mock.clone();

    // @step When the user runs `laya serve --max-model-len 4096`
    // (the flag is passed straight through `ServerConfig`; the clamp is the
    //  `ServerState` seam (JEV-005) this card wires up — the flag's own
    //  parse-time acceptance is exercised by the parse-time tests)
    let state = ServerState::new(
        answerer,
        "checkpoint",
        ServerConfig { max_model_len: Some(4096), ..Default::default() },
    );

    // @step Then the effective cap used for pre-inference admission is 1024 (the smaller of the flag and the checkpoint's native limit)
    state
        .execute(json!({ "subject": "Duplicate charge" }), vec![("q".to_string(), one_choice_question())])
        .await
        .expect("execute");
    assert_eq!(
        *mock.admit_cap.lock().expect("admit_cap poisoned"),
        Some(1024),
        "--max-model-len 4096 must clamp to the checkpoint's native max_len 1024"
    );
}

/// Scenario: The supplied model value is the reported model identity
#[test]
fn the_supplied_model_value_is_the_reported_model_identity() {
    // @step Given the server started with `--model-variant typed-decisions` (or an explicit --model path)
    // (LAYA_TEST_MODEL as the explicit --model path; the variant-key form
    //  reports the key through the identical health/identity seam — the
    //  weight-free start_server test covers that path with the mock)
    let model = match model_dir() {
        Some(dir) => dir,
        None => return, // skipped: no checkpoint available (LAYA_TEST_MODEL unset)
    };
    let port = free_port();
    let mut child = spawn_serve(&model, port);

    // @step When /health is polled and a request with a different model field is POSTed
    let health_body = http(&format!("127.0.0.1:{port}"), "GET", "/health", None)
        .expect("reachable after wait_for_health")
        .1;
    let (status, resp) = http(&format!("127.0.0.1:{port}"), "POST", "/v1/classifier", Some(&classifier_body("some-other-model")))
        .expect("classifier POST");

    // @step Then /health reports the supplied value and the mismatched request is rejected with 422 naming the loaded model
    assert!(
        health_body.contains(&model),
        "health must report the supplied value: {health_body}"
    );
    assert_eq!(status, 422, "the mismatched request must be a 422; body: {resp}");
    assert!(resp.contains("Loaded model is"), "the 422 must name the loaded model: {resp}");
    assert!(resp.contains(&model), "the 422 must name the loaded model: {resp}");

    let _ = child.kill();
    let _ = child.wait();
}

/// Scenario: Ctrl-C stops the server gracefully
#[ntex::test]
async fn ctrl_c_stops_the_server_gracefully() {
    // @step Given the server is running with a long in-flight forward
    // (weight-free stand-in: a 400ms mock forward with a request in flight
    //  when the shutdown is triggered; the binary's Ctrl-C/SIGTERM path is
    //  the same `handle.stop()` seam, exercised in the model-gated half)
    let mock = Arc::new(MockAnswerer::new((1024, 128), 400));
    let answerer: Arc<dyn Answerer> = mock.clone();
    let handle = start_server("127.0.0.1", 0, answerer, "graceful-model", ServerConfig::default())
        .await
        .expect("start_server");
    let host_port = format!("127.0.0.1:{}", handle.port);

    // The in-flight forward: the request's mock forward is running when the
    // shutdown arrives (it started at t=0, runs 400ms; the shutdown lands at
    // ~150ms).
    let body = classifier_body("graceful-model");
    let host_port2 = host_port.clone();
    let task = ntex::rt::spawn_blocking(move || http(&host_port2, "POST", "/v1/classifier", Some(&body)));
    ntex::time::sleep(ntex::time::Millis(150)).await;

    // @step When the user presses Ctrl-C (or sends SIGTERM)
    handle.stop().await; // the binary maps Ctrl-C/SIGTERM to this same stop()

    // @step Then the in-flight forward completes, the listener stops accepting new requests, and the process exits 0
    let (status, resp) = task
        .await
        .expect("client task must complete")
        .expect("the in-flight request must complete");
    assert_eq!(status, 200, "the in-flight forward must complete: {resp}");
    assert!(
        mock.forward_completed.load(Ordering::SeqCst),
        "the in-flight forward must have run to completion"
    );
    assert!(
        http(&host_port, "GET", "/health", None).is_none(),
        "the listener must stop accepting new requests after stop"
    );

    // Model-gated half: the real process exits 0 on SIGTERM.
    let model = match model_dir() {
        Some(dir) => dir,
        None => return, // skipped: no checkpoint available (LAYA_TEST_MODEL unset)
    };
    let port = free_port();
    let mut child = spawn_serve(&model, port);
    let _ = http(&format!("127.0.0.1:{port}"), "GET", "/health", None);
    Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .expect("send SIGTERM");
    let status = child.wait().expect("wait for serve child");
    assert!(status.success(), "SIGTERM must stop the server with exit code 0");
}

/// Scenario: Invalid serve flags are rejected at parse time
#[test]
fn invalid_serve_flags_are_rejected_at_parse_time() {
    // @step Given the flags --max-queued 0 or --max-request-branches -1
    // @step When the user runs `laya serve`
    for (flag, value) in [("--max-queued", "0"), ("--max-request-branches", "-1")] {
        let out = Command::new(laya_bin())
            .args(["serve", flag, value])
            .output()
            .expect("run laya serve");

        // @step Then clap rejects the flags with a usage error before any checkpoint is loaded
        assert!(!out.status.success(), "{flag} {value} must be rejected at parse time");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains(flag), "the usage error must name the flag: {stderr}");
        assert!(!stderr.contains("laya serving"), "nothing may be served: {stderr}");
        assert!(!stderr.contains("loading"), "no checkpoint may be loaded: {stderr}");
    }
}
