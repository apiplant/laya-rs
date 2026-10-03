/**
 * Feature: spec/features/laya-native-answer-mapping-usage-accounting-serial-execution-429-admission-queue.feature
 *
 * This test file validates the acceptance criteria defined in the feature file.
 * Scenarios map directly to Gherkin scenarios.
 *
 * All scenarios are weight-free: the answer mapping is pure over hand-built
 * `laya::Answer` values; execution/queue semantics run against a mock
 * `Answerer` (no checkpoint, no GPU); and the sequence admission is
 * exercised against the real `schema::build_sequence` with the in-memory
 * WordLevel tokenizer pattern from tests/schema.rs (special tokens:
 * [unk]=0, [PAD]=1, [CLS]=2, [SEP]=3, [MASK]=4; words added from 5).
 */

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use laya::schema::{serialize_state, QType, Question, SpecialTokens};
use laya::server::response::{answer_to_jev_json, ClassifierResponse, Usage};
use laya::server::state::{
    admit_questions, AdmissionError, Answerer, ExecutionError, ServerConfig, ServerState,
};
use laya::Answer;
use serde_json::json;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

/// In-memory WordLevel tokenizer with a Whitespace pre-tokenizer (tests/schema.rs pattern).
fn test_tokenizer(words: &[&str]) -> Tokenizer {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("<unk>".to_string(), 0);
    vocab.insert("[PAD]".to_string(), 1);
    vocab.insert("[CLS]".to_string(), 2);
    vocab.insert("[SEP]".to_string(), 3);
    vocab.insert("[MASK]".to_string(), 4);
    for (i, w) in words.iter().enumerate() {
        vocab.insert(w.to_string(), 5 + i as u32);
    }
    let model = WordLevel::builder().vocab(vocab).build().expect("word-level model");
    let mut tok = Tokenizer::new(model);
    tok.with_pre_tokenizer(Some(Whitespace::default()));
    tok
}

fn special() -> SpecialTokens {
    SpecialTokens {
        cls: "[CLS]".to_string(),
        sep: "[SEP]".to_string(),
        mask: "[MASK]".to_string(),
        pad: "[PAD]".to_string(),
    }
}

fn noul_q(id: &str) -> (String, Question) {
    (
        id.to_string(),
        Question {
            qtype: QType::Noul,
            instructions: "Does it hold?".to_string(),
            choice_criteria: vec![],
            score_criteria: vec![],
            noul_true: None,
            noul_false: None,
        },
    )
}

/// A std-sync gate: `wait()` blocks a (blocking-pool) thread until `open()`.
/// Keeps the queue tests fully deterministic on a single-threaded runtime —
/// the mock forward parks here instead of sleeping a fixed duration.
struct Gate {
    cond: std::sync::Condvar,
    open: std::sync::Mutex<bool>,
}

impl Gate {
    fn new() -> Self {
        Self { cond: std::sync::Condvar::new(), open: std::sync::Mutex::new(false) }
    }
    fn wait(&self) {
        let mut open = self.open.lock().expect("gate poisoned");
        while !*open {
            open = self.cond.wait(open).expect("gate poisoned");
        }
    }
    fn open(&self) {
        let mut open = self.open.lock().expect("gate poisoned");
        *open = true;
        self.cond.notify_all();
    }
}

/// Weight-free `Answerer` with configurable behavior + instrumentation.
#[derive(Default)]
struct MockAnswerer {
    /// When set, each question's synthetic sequence length is
    /// `fixed_lengths[i]` (per position), used for the usage-accounting test.
    fixed_lengths: Vec<usize>,
    /// When set, each question's synthetic length is `4 + serialized state len`.
    lengths_from_state: bool,
    /// When set, `system_one` parks at this gate before answering (a
    /// deterministic "slow in-flight forward" without sleeping).
    gate: Option<Arc<Gate>>,
    fail_once: AtomicBool,
    calls: AtomicUsize,
    current: AtomicUsize,
    max_concurrent: AtomicUsize,
}

impl MockAnswerer {
    fn forwards(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
    fn max_concurrent(&self) -> usize {
        self.max_concurrent.load(Ordering::SeqCst)
    }
}

impl Answerer for MockAnswerer {
    fn checkpoint_limits(&self) -> (usize, usize) {
        (1024, 128)
    }

    fn admit(
        &self,
        state: &serde_json::Value,
        questions: &[(String, Question)],
        cap: usize,
    ) -> Result<Vec<usize>, AdmissionError> {
        let mut out = Vec::with_capacity(questions.len());
        for (i, (qid, _)) in questions.iter().enumerate() {
            let len = if self.lengths_from_state {
                4 + serialize_state(state).len()
            } else if let Some(l) = self.fixed_lengths.get(i) {
                *l
            } else {
                32
            };
            if len > cap {
                return Err(AdmissionError::ExceedsTokenCap { question: qid.clone(), cap });
            }
            out.push(len);
        }
        Ok(out)
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
        let before = self.current.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_concurrent.fetch_max(before, Ordering::SeqCst);
        if let Some(gate) = &self.gate {
            gate.wait(); // deterministic "slow forward": parked until the test releases it
        }
        self.current.fetch_sub(1, Ordering::SeqCst);
        Ok(questions
            .iter()
            .map(|(qid, _)| (qid.clone(), Answer::Noul { noul: 0.5, act_probability: 0.0 }))
            .collect())
    }
}

/// Scenario: A mixed request maps to laya-native protocol answers
#[test]
fn a_mixed_request_maps_to_laya_native_protocol_answers() {
    // @step Given a completed mixed request with a choice (billing/technical), a score (3 levels), and a noul question
    let route = Answer::Choice {
        choice: "billing".to_string(),
        probabilities: vec![("billing".to_string(), 0.8), ("technical".to_string(), 0.2)],
        confidence: 0.8,
        act_probability: 0.9, // must be stripped from the protocol answer
    };
    let urgency = Answer::Score {
        score: 1.75,
        legend: vec!["Routine".to_string(), "Important".to_string(), "Critical".to_string()],
        probabilities: vec![0.1, 0.1, 0.8],
        confidence: 0.8,
        act_probability: 0.5,
    };
    let refund = Answer::Noul { noul: 0.9, act_probability: 0.4 };

    // @step When the answers are mapped to the Jev protocol response shape
    let response = ClassifierResponse {
        model: "convaiinnovations/laya".to_string(),
        answers: vec![
            ("route".to_string(), answer_to_jev_json(&route)),
            ("urgency".to_string(), answer_to_jev_json(&urgency)),
            ("refund".to_string(), answer_to_jev_json(&refund)),
        ],
        usage: Usage { input_tokens: 600, output_tokens: 0 },
    };
    let text = response.to_json();
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("protocol JSON must parse");

    // @step Then answers.route is {type: choice, choice: 'billing', confidence, probabilities keyed in request order billing/technical} with no act_probability field
    let r = &parsed["answers"]["route"];
    assert_eq!(r["type"], "choice");
    assert_eq!(r["choice"], "billing");
    assert_eq!(r["confidence"].as_f64(), Some(0.8));
    assert_eq!(r["probabilities"]["billing"].as_f64(), Some(0.8));
    assert_eq!(r["probabilities"]["technical"].as_f64(), Some(0.2));
    assert_eq!(r["probabilities"].as_object().map(|o| o.len()), Some(2));
    assert!(!text.contains("act_probability"), "act_probability must be stripped from every answer");

    // @step And answers.urgency is {type: score, score: 1.75, confidence: 0.8, probabilities {"0":..,"1":..,"2":..}, legend {"0":"Routine",...}}
    let u = &parsed["answers"]["urgency"];
    assert_eq!(u["type"], "score");
    assert_eq!(u["score"].as_f64(), Some(1.75));
    assert_eq!(u["confidence"].as_f64(), Some(0.8));
    assert_eq!(u["probabilities"]["0"].as_f64(), Some(0.1));
    assert_eq!(u["probabilities"]["1"].as_f64(), Some(0.1));
    assert_eq!(u["probabilities"]["2"].as_f64(), Some(0.8));
    assert_eq!(u["legend"]["0"], "Routine");
    assert_eq!(u["legend"]["1"], "Important");
    assert_eq!(u["legend"]["2"], "Critical");

    // @step And answers.refund is {type: noul, noul: 0.9} with no confidence field
    let n = &parsed["answers"]["refund"];
    assert_eq!(n["type"], "noul");
    assert_eq!(n["noul"].as_f64(), Some(0.9));
    assert!(n.get("confidence").is_none(), "noul must carry no confidence field");
    assert_eq!(n.as_object().map(|o| o.len()), Some(2), "noul carries ONLY type + noul");

    // Request insertion order (not BTreeMap-sorted order) must be preserved in
    // the serialized probabilities: a non-alphabetical request order zulu, alpha
    // must serialize zulu first (a sorted map would emit alpha first).
    let reversed = Answer::Choice {
        choice: "zulu".to_string(),
        probabilities: vec![("zulu".to_string(), 0.6), ("alpha".to_string(), 0.4)],
        confidence: 0.6,
        act_probability: 0.0,
    };
    let rt = ClassifierResponse {
        model: "m".to_string(),
        answers: vec![("q".to_string(), answer_to_jev_json(&reversed))],
        usage: Usage { input_tokens: 1, output_tokens: 0 },
    }
    .to_json();
    let p = rt.find("probabilities").expect("probabilities object present");
    let after = &rt[p..];
    let z = after.find("\"zulu\"").expect("zulu present in probabilities");
    let a = after.find("\"alpha\"").expect("alpha present in probabilities");
    assert!(z < a, "probabilities must keep the request's insertion order (zulu before alpha), got: {rt}");
}

/// Scenario: Usage accounting sums per-question sequence lengths
#[ntex::test]
async fn usage_accounting_sums_per_question_sequence_lengths() {
    // @step Given the mixed three-question request above
    let server = ServerState::new(
        Arc::new(MockAnswerer { fixed_lengths: vec![100, 200, 300], ..Default::default() }),
        "convaiinnovations/laya",
        ServerConfig::default(),
    );
    let questions = vec![noul_q("route"), noul_q("urgency"), noul_q("refund")];

    // @step When the response usage is computed
    let response = server
        .execute(json!("We were billed twice for March."), questions)
        .await
        .expect("execution must succeed");

    // @step Then usage.input_tokens equals the sum of the three per-question tokenized sequence lengths (the shared state text is counted three times)
    assert_eq!(
        response.usage.input_tokens, 600,
        "usage.input_tokens must sum the per-question sequence lengths (100+200+300)"
    );

    // @step And usage.output_tokens is 0
    assert_eq!(response.usage.output_tokens, 0);
}

/// Scenario: Oversized questions are rejected before inference
#[ntex::test]
async fn oversized_questions_are_rejected_before_inference() {
    // @step Given a request whose one question's compiled sequence exceeds the effective max-model-len cap
    let long_words: Vec<String> = (0..60).map(|i| format!("word{i}")).collect();
    let base: Vec<&str> = vec![
        "choice", "question:", "Which", "one", "of", "the", "following", "many", "numbered",
        "items", "applies", "here?", "first", "second",
    ];
    let words: Vec<&str> = base
        .into_iter()
        .chain(long_words.iter().map(|s| s.as_str()))
        .collect();
    let tok = test_tokenizer(&words);

    let q = Question {
        qtype: QType::Choice,
        instructions: "Which one of the following many numbered items applies here?".to_string(),
        choice_criteria: vec![("first".to_string(), None), ("second".to_string(), None)],
        score_criteria: vec![],
        noul_true: None,
        noul_false: None,
    };
    // 60 words of state (+ head/options) cannot fit the 32-token cap.
    let state = json!(long_words.join(" "));

    // @step When the request is admitted
    let res = admit_questions(&tok, &special(), &state, &[("q1".to_string(), q.clone())], 32, 64);
    // @step Then the response is 422 naming the question id and the cap
    match res {
        Err(AdmissionError::ExceedsTokenCap { question, cap }) => {
            assert_eq!(question, "q1", "the 422 must name the question id");
            assert_eq!(cap, 32, "the 422 must name the effective cap");
            let msg = AdmissionError::ExceedsTokenCap { question, cap }.message();
            assert!(msg.contains("q1"), "message must name the question id: {msg}");
            assert!(msg.contains("32"), "message must name the cap: {msg}");
        }
        other => panic!("an over-cap compiled sequence must be rejected before inference, got {other:?}"),
    }

    // @step And no forward pass was executed (the model was never called)
    // Admission is the pre-inference gate: `execute` must stop here and never
    // reach the answerer's `system_one`.
    let mock = Arc::new(MockAnswerer { lengths_from_state: true, ..Default::default() });
    let server = ServerState::new(
        mock.clone(),
        "m",
        ServerConfig { max_model_len: Some(8), ..Default::default() },
    );
    let err = server
        .execute(json!("some state text"), vec![noul_q("q1")])
        .await
        .expect_err("an over-cap request must be rejected");
    match err {
        ExecutionError::Admission(AdmissionError::ExceedsTokenCap { question, cap }) => {
            assert_eq!(question, "q1");
            assert_eq!(cap, 8, "the effective cap is min(--max-model-len, checkpoint max_len)");
        }
        other => panic!("expected an over-cap admission error, got {other:?}"),
    }
    assert_eq!(mock.forwards(), 0, "the model must never be called for a rejected request");
}

/// Scenario: Excess question count is rejected before inference
#[ntex::test]
async fn excess_question_count_is_rejected_before_inference() {
    // @step Given a request with more questions than --max-request-branches (100)
    assert_eq!(
        ServerConfig::default().max_request_branches, 100,
        "the --max-request-branches default must be 100"
    );
    let mock = Arc::new(MockAnswerer::default());
    let server = ServerState::new(
        mock.clone(),
        "m",
        ServerConfig { max_request_branches: 2, ..Default::default() },
    );
    let questions = vec![noul_q("q1"), noul_q("q2"), noul_q("q3")];

    // @step When the request is admitted
    let err = server
        .execute(json!("s"), questions.clone())
        .await
        .expect_err("an over-branch request must be rejected");

    // @step Then the response is 422 naming the branch limit and no forward pass was executed
    match err {
        ExecutionError::Admission(AdmissionError::TooManyQuestions { count, limit }) => {
            assert_eq!((count, limit), (3, 2));
            let msg = AdmissionError::TooManyQuestions { count: 3, limit: 2 }.message();
            assert!(msg.contains("2"), "message must name the branch limit: {msg}");
        }
        other => panic!("expected a TooManyQuestions admission error, got {other:?}"),
    }
    assert_eq!(mock.forwards(), 0, "the model must never be called for a rejected request");
}

/// Scenario: A full admission queue yields 429 with Retry-After
#[ntex::test]
async fn a_full_admission_queue_yields_429() {
    // @step Given the server is configured with --max-queued 1 and one slow forward is in flight
    let gate = Arc::new(Gate::new());
    let mock = Arc::new(MockAnswerer { gate: Some(gate.clone()), ..Default::default() });
    let server = ServerState::new(
        mock.clone(),
        "m",
        ServerConfig { max_queued: 1, ..Default::default() },
    );
    let questions = vec![noul_q("q1")];

    // @step When a third concurrent request arrives
    let s1 = server.clone();
    let q1 = questions.clone();
    let e1 = ntex::rt::spawn(async move { s1.execute(json!("s"), q1).await });
    // Deterministic: e1 holds its in-flight permit (parked inside the forward,
    // gate closed). `admission_available_slots` is observable on the runtime
    // thread, unlike the forward's internals (blocking thread).
    while server.admission_available_slots() > 1 {
        ntex::time::sleep(ntex::time::Millis(1)).await;
    }
    let s2 = server.clone();
    let q2 = questions.clone();
    let e2 = ntex::rt::spawn(async move { s2.execute(json!("s"), q2).await }); // takes the single waiting slot
    // e2 is parked on the model lock, holding its waiting permit:
    while server.admission_available_slots() > 0 {
        ntex::time::sleep(ntex::time::Millis(1)).await;
    }
    let s3 = server.clone();
    let q3 = questions.clone();
    let e3 = s3.execute(json!("s"), q3).await;

    // @step Then it receives 429 {"detail":"Scoring queue is full"} with header Retry-After: 1
    // (the 429 status + Retry-After envelope is built by the router error layer,
    // JEV-004; at the execution seam the full queue surfaces as QueueFull)
    match e3 {
        Err(ExecutionError::QueueFull) => {}
        other => panic!("a request arriving at capacity must be rejected with QueueFull, got {other:?}"),
    }

    // @step And the first two requests proceed and exactly one forward is ever running at a time (verified with an instrumented mock answerer)
    gate.open(); // release the in-flight forward; e2 then proceeds serially behind it
    let r1 = e1.await.expect("first request task must not panic").expect("first request must succeed");
    let r2 = e2.await.expect("second request task must not panic").expect("second request must succeed");
    assert_eq!(r1.usage.output_tokens, 0);
    assert_eq!(r2.usage.input_tokens, 32);
    assert_eq!(mock.max_concurrent(), 1, "exactly one forward may run at a time (serial execution)");
}

/// Scenario: Forward errors and disconnects do not corrupt the queue
#[ntex::test]
async fn forward_errors_and_disconnects_do_not_corrupt_the_queue() {
    // @step Given a forward that raises an internal error, or a client that disconnects while queued
    // (a) internal forward error:
    let mock = Arc::new(MockAnswerer { fail_once: AtomicBool::new(true), ..Default::default() });
    let server = ServerState::new(mock.clone(), "m", ServerConfig::default());
    let questions = vec![noul_q("q1")];
    let r1 = server.clone().execute(json!("s"), questions.clone()).await;

    // @step When the request completes (or is cancelled)
    // @step Then the model lock is released (the in-flight forward runs to completion), subsequent requests are served, and a disconnect that can still be answered is reported as 499 {"detail":"Client disconnected"}
    // (the 499 body is the router error layer's, JEV-004; at the execution seam
    // the invariants are: the lock is released after a failed forward and the
    // server keeps serving)
    match r1 {
        Err(ExecutionError::Internal(e)) => {
            let msg = e.to_string();
            assert!(msg.contains("simulated internal forward error"), "unexpected error: {msg}");
        }
        other => panic!("a failing forward must surface as Internal, got {other:?}"),
    }
    let r2 = server.clone().execute(json!("s"), questions.clone()).await;
    assert!(r2.is_ok(), "the server must keep serving after an internal forward error: {r2:?}");

    // (b) a client that disconnects while queued (its queued future is cancelled):
    let gate = Arc::new(Gate::new());
    let mock2 = Arc::new(MockAnswerer { gate: Some(gate.clone()), ..Default::default() });
    let server2 = ServerState::new(
        mock2.clone(),
        "m",
        ServerConfig { max_queued: 1, ..Default::default() },
    );
    let s1 = server2.clone();
    let q1 = questions.clone();
    let e1 = ntex::rt::spawn(async move { s1.execute(json!("s"), q1).await });
    // e1 holds its in-flight permit, parked inside the forward (gate closed).
    while server2.admission_available_slots() > 1 {
        ntex::time::sleep(ntex::time::Millis(1)).await;
    }
    let s2 = server2.clone();
    let q2 = questions.clone();
    let e2 = ntex::rt::spawn(async move { s2.execute(json!("s"), q2).await }); // queued (holding the waiting slot)
    // While e1 is in flight and e2 is waiting, the queue is full:
    while server2.admission_available_slots() > 0 {
        ntex::time::sleep(ntex::time::Millis(1)).await;
    }
    let s3 = server2.clone();
    let q3 = questions.clone();
    let probe = s3.execute(json!("s"), q3).await;
    assert!(matches!(probe, Err(ExecutionError::QueueFull)), "1 in-flight + 1 waiting must be full: {probe:?}");

    // @step When the request completes (or is cancelled)
    e2.cancel(); // the client disconnects while queued: its future (and permit) is cancelled
    gate.open(); // the in-flight forward runs to completion — it cannot be interrupted

    // @step Then the model lock is released (the in-flight forward runs to completion), subsequent requests are served, and a disconnect that can still be answered is reported as 499 {"detail":"Client disconnected"}
    e1.await.expect("in-flight forward must run to completion").expect("first request must succeed");
    let s4 = server2.clone();
    let q4 = questions.clone();
    let e3 = s4.execute(json!("s"), q4).await;
    e3.expect("a subsequent request must be served after a cancelled queued request");
    assert_eq!(mock2.forwards(), 2, "only the non-cancelled requests may forward (e1 + e3, not e2)");
}
