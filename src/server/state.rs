//! Server execution layer (JEV-005): the [`Answerer`] seam, pre-inference
//! admission, serial execution, and the bounded 429 admission queue.
//!
//! The protocol contract (JEV-002): requests execute **serially** against the
//! model (one forward at a time; all parallelism is inside a request's single
//! `system_one` call), with one in-flight forward plus up to
//! `--max-queued` waiting requests. A request arriving at capacity is rejected
//! with 429 `{"detail":"Scoring queue is full"}` + `Retry-After: 1` (the router
//! error layer, JEV-004, renders this seam's [`ExecutionError::QueueFull`]).
//!
//! `ServerState` is the `Clone`-over-`Arc` shared state the ntex routes
//! (`build_router`, JEV-004) extracts; the `Answerer` trait is the seam that
//! keeps the whole protocol layer testable without a checkpoint — production
//! installs [`RealAnswerer`] over the loaded [`laya::RLAgent`], weight-free
//! tests inject mocks.

use std::sync::Arc;

use serde_json::Value;
use tokenizers::Tokenizer;

use crate::agent::RLAgent;
use crate::schema::{build_sequence, render_options, Question, SpecialTokens};
use crate::Answer;

use super::response::{answer_to_jev_json, ClassifierResponse, Usage};

/// Runtime limits for the classifier endpoint (protocol reference,
/// "Server startup arguments (reference CLI, for parity)").
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// `--max-model-len`: effective per-question sequence cap. `None` means
    /// "use the checkpoint's native `max_len`" (the reference default).
    pub max_model_len: Option<usize>,
    /// `--max-request-branches`: max questions per request (default 100; the
    /// schema hard cap of 256 is enforced by the request schema itself).
    pub max_request_branches: usize,
    /// Admission queue: waiting slots *on top of* the one in-flight forward
    /// (the reference's 1+16 model; default 16).
    pub max_queued: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self { max_model_len: None, max_request_branches: 100, max_queued: 16 }
    }
}

/// The inference seam: one `system_one` over all of a request's questions,
/// plus the tokenizer access admission needs to compile question sequences.
///
/// `Send + Sync` so the trait object lives behind `Arc` in server state and
/// the blocking forward can be offloaded to the blocking thread pool.
pub trait Answerer: Send + Sync {
    /// The checkpoint's native limits: `(max_len, head_max_len)`.
    fn checkpoint_limits(&self) -> (usize, usize);

    /// Compile every question's sequence against the effective cap, *before*
    /// any forward pass. Ok carries one length per question (in request
    /// order) — the same numbers `usage.input_tokens` sums; Err is a 422
    /// admission failure naming the offending question and the cap/limit.
    fn admit(
        &self,
        state: &Value,
        questions: &[(String, Question)],
        effective_cap: usize,
    ) -> Result<Vec<usize>, AdmissionError>;

    /// Run the model over all questions (one batched forward). Synchronous;
    /// the server runs it under `ntex::rt::spawn_blocking`.
    fn system_one(&self, state: &Value, questions: &[(String, Question)]) -> anyhow::Result<Vec<(String, Answer)>>;
}

/// A pre-inference admission failure — a 422 with no forward ever executed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionError {
    /// One question's compiled sequence exceeds the effective cap
    /// (`min(--max-model-len, checkpoint max_len)`); no truncation is applied.
    ExceedsTokenCap { question: String, cap: usize },
    /// A question's option markers were lost when compiling against the cap.
    MarkerLoss { question: String },
    /// More questions than `--max-request-branches` allow.
    TooManyQuestions { count: usize, limit: usize },
}

impl AdmissionError {
    /// The 422 error message (the router layer wraps it in the protocol
    /// envelope, JEV-004).
    pub fn message(&self) -> String {
        match self {
            AdmissionError::ExceedsTokenCap { question, cap } => {
                format!("question '{question}' exceeds {cap} input tokens")
            }
            AdmissionError::MarkerLoss { question } => {
                format!("question '{question}': options do not fit within the model length cap")
            }
            AdmissionError::TooManyQuestions { count, limit } => {
                format!("{count} questions exceed the {limit}-branch request limit")
            }
        }
    }
}

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message())
    }
}

impl std::error::Error for AdmissionError {}

/// A model forward failure (`anyhow::Error` does not implement `std::error::Error`,
/// so it is boxed for the `source()` chain).
#[derive(Debug)]
pub struct InternalError {
    inner: anyhow::Error,
}

impl std::fmt::Display for InternalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

impl std::error::Error for InternalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // `anyhow::Error` is not itself a `dyn Error`; surface its context as
        // the message instead.
        None
    }
}

/// An execution-layer failure surfaced by [`ServerState::execute`].
#[derive(Debug)]
pub enum ExecutionError {
    /// A 422 admission failure (no forward pass executed).
    Admission(AdmissionError),
    /// The admission queue was full: the router renders this as 429
    /// `{"detail":"Scoring queue is full"}` with `Retry-After: 1`.
    QueueFull,
    /// The model forward failed: the router renders this as 500.
    Internal(InternalError),
}

impl std::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionError::Admission(e) => write!(f, "{e}"),
            ExecutionError::QueueFull => f.write_str("Scoring queue is full"),
            ExecutionError::Internal(e) => write!(f, "internal error: {e}"),
        }
    }
}

impl std::error::Error for ExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ExecutionError::Admission(e) => Some(e),
            ExecutionError::QueueFull => None,
            ExecutionError::Internal(e) => Some(e),
        }
    }
}

impl From<AdmissionError> for ExecutionError {
    fn from(e: AdmissionError) -> Self {
        ExecutionError::Admission(e)
    }
}

/// A cap far above any model length (the largest checkpoint `max_len` is
/// 16384): used only to measure the *natural*, untruncated sequence length.
/// (Not `usize::MAX` — `build_sequence` reserves a `Vec` of that capacity.)
const NATURAL_CAP: usize = 1_000_000;

/// Compile one question at the given limits and return the full (untruncated)
/// length, the marker positions at the cap, and the number of option markers.
fn admit_question(
    tok: &Tokenizer,
    special: &SpecialTokens,
    state: &Value,
    q: &Question,
    cap: usize,
    head_max_len: usize,
) -> (usize, Vec<usize>, usize) {
    // Measure the natural (untruncated) length: build with a far-generous cap
    // so nothing is truncated, then compare it against the protocol cap.
    let natural = build_sequence(tok, special, state, q, NATURAL_CAP, NATURAL_CAP);
    let at_cap = build_sequence(tok, special, state, q, cap, head_max_len);
    let n_opts = render_options(q).len();
    (natural.ids.len(), at_cap.markers, n_opts)
}

/// Pre-inference admission over a whole request (see [`Answerer::admit`]).
/// Pure over a tokenizer — the weight-free tests drive this directly.
/// Returns one per-question natural sequence length, in request order.
pub fn admit_questions(
    tok: &Tokenizer,
    special: &SpecialTokens,
    state: &Value,
    questions: &[(String, Question)],
    effective_cap: usize,
    head_max_len: usize,
) -> Result<Vec<usize>, AdmissionError> {
    let mut lengths = Vec::with_capacity(questions.len());
    for (qid, q) in questions {
        let (natural_len, markers_at_cap, n_opts) = admit_question(tok, special, state, q, effective_cap, head_max_len);
        if markers_at_cap.len() != n_opts {
            return Err(AdmissionError::MarkerLoss { question: qid.clone() });
        }
        if natural_len > effective_cap {
            return Err(AdmissionError::ExceedsTokenCap { question: qid.clone(), cap: effective_cap });
        }
        lengths.push(natural_len);
    }
    Ok(lengths)
}

/// The production answerer: forwards to the loaded [`RLAgent`].
pub struct RealAnswerer {
    agent: Arc<RLAgent>,
}

impl RealAnswerer {
    pub fn new(agent: Arc<RLAgent>) -> Self {
        Self { agent }
    }
}

impl Answerer for RealAnswerer {
    fn checkpoint_limits(&self) -> (usize, usize) {
        self.agent.checkpoint_limits()
    }

    fn admit(
        &self,
        state: &Value,
        questions: &[(String, Question)],
        effective_cap: usize,
    ) -> Result<Vec<usize>, AdmissionError> {
        let head_max_len = self.agent.checkpoint_limits().1;
        admit_questions(
            self.agent.tokenizer(),
            self.agent.special_tokens(),
            state,
            questions,
            effective_cap,
            head_max_len,
        )
    }

    fn system_one(&self, state: &Value, questions: &[(String, Question)]) -> anyhow::Result<Vec<(String, Answer)>> {
        self.agent.system_one(state, questions)
    }
}

/// Shared server state: the ntex `State` extractor for the classifier handler
/// (JEV-004). `Clone` is cheap — it wraps an `Arc`.
///
/// Concurrency: exactly one model forward runs at a time
/// (`model_lock: tokio::sync::Mutex<()>`, held across the blocking offload so
/// admission ordering is FIFO-ish), and `admission: Semaphore` bounds the
/// in-flight + waiting population (the reference's 1-in-flight + `max_queued`
/// waiting model).
#[derive(Clone)]
pub struct ServerState {
    inner: Arc<Inner>,
}

struct Inner {
    answerer: Arc<dyn Answerer>,
    pub model_name: String,
    pub config: ServerConfig,
    // `Arc`-wrapped so the `*_owned` accessors (`lock_owned`,
    // `try_acquire_owned`) can be used directly on the shared state.
    model_lock: Arc<tokio::sync::Mutex<()>>,
    admission: Arc<tokio::sync::Semaphore>,
}

impl ServerState {
    pub fn new(answerer: Arc<dyn Answerer>, model_name: impl Into<String>, config: ServerConfig) -> Self {
        let max_queued = config.max_queued;
        Self {
            inner: Arc::new(Inner {
                answerer,
                model_name: model_name.into(),
                config,
                model_lock: Arc::new(tokio::sync::Mutex::new(())),
                // One permit for the in-flight forward + one per waiting slot.
                admission: Arc::new(tokio::sync::Semaphore::new(max_queued.saturating_add(1))),
            }),
        }
    }

    pub fn model_name(&self) -> &str {
        &self.inner.model_name
    }

    /// Admission slots currently free (the in-flight forward + all waiting
    /// requests combined hold the rest). `0` means the queue is full — a new
    /// request would be rejected with `QueueFull` (429). Exposed for
    /// observability/diagnostics.
    pub fn admission_available_slots(&self) -> usize {
        self.inner.admission.available_permits()
    }

    pub fn config(&self) -> &ServerConfig {
        &self.inner.config
    }

    /// The effective per-question sequence cap:
    /// `min(--max-model-len, checkpoint max_len)` (`None` flag ⇒ checkpoint
    /// native `max_len`).
    fn effective_cap(&self) -> usize {
        let native = self.inner.answerer.checkpoint_limits().0;
        self.inner.config.max_model_len.map(|c| c.min(native)).unwrap_or(native)
    }

    /// Run one classifier request end-to-end: pre-inference admission (422
    /// before any forward), then the bounded admission queue (429 when full),
    /// then the serial forward under the model lock, offloaded to the
    /// blocking pool. Returns the assembled protocol response.
    ///
    /// Takes the context and questions by value (the handler owns them); a
    /// `Clone` of [`ServerState`] plus the owned payload is all a spawned
    /// request task needs — which is also how client-disconnect cancellation
    /// works (the router, JEV-004/006): when the client goes away ntex drops
    /// the task's future; a cancelled *queued* request releases its
    /// admission permit (the semaphore's guard does that on drop) and an
    /// in-flight forward runs to completion — it cannot be interrupted
    /// (reference behavior).
    pub async fn execute(
        &self,
        state: Value,
        questions: Vec<(String, Question)>,
    ) -> Result<ClassifierResponse, ExecutionError> {
        let cap = self.effective_cap();
        let max_branches = self.inner.config.max_request_branches;

        // 1. Pre-inference admission — 422 before any forward pass. The
        //    per-question lengths computed here are exactly what
        //    `usage.input_tokens` sums (no double tokenization).
        if questions.len() > max_branches {
            return Err(ExecutionError::Admission(AdmissionError::TooManyQuestions {
                count: questions.len(),
                limit: max_branches,
            }));
        }
        let lengths = self.inner.answerer.admit(&state, &questions, cap).map_err(ExecutionError::Admission)?;

        // 2. Admission queue: one in-flight forward + `max_queued` waiting.
        //    Full ⇒ 429 (the router renders `{"detail":"Scoring queue is
        //    full"}` + `Retry-After: 1`). Both the permit and the lock guard
        //    live until this function returns, so a *dropped* request future
        //    (client disconnect / test `abort`) releases its waiting slot
        //    automatically — an in-flight forward, already offloaded to the
        //    blocking pool, keeps running to completion (it cannot be
        //    interrupted — reference behavior).
        let permit = match self.inner.admission.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => return Err(ExecutionError::QueueFull),
        };

        // 3. Serial execution: exactly one forward at a time. Held across the
        //    blocking offload, so admission ordering is FIFO-ish and no
        //    second forward can start while one is running.
        let lock = self.inner.model_lock.clone().lock_owned().await;

        // 4. The candle forward is synchronous — run it on the blocking pool,
        //    never on an async runtime thread.
        let answerer = Arc::clone(&self.inner.answerer);
        //    The permit and the lock guard move *into* the closure: if the client
        //    disconnects and this future is dropped mid-forward, the forward keeps
        //    running on the blocking pool and the lock/slot are released only when
        //    it actually finishes — so a second forward can never overlap it.
        let result = ntex::rt::spawn_blocking(move || {
            let answers = answerer.system_one(&state, &questions);
            drop(lock);
            drop(permit);
            answers
        })
        .await;

        let answers = match result {
            Ok(Ok(answers)) => answers,
            Ok(Err(e)) => return Err(ExecutionError::Internal(InternalError { inner: e })),
            Err(join) => {
                return Err(ExecutionError::Internal(InternalError {
                    inner: anyhow::anyhow!("forward task failed: {join}"),
                }))
            }
        };

        // 5. Protocol response assembly (JEV-005 mapping). The lock guard and
        //    permit drop here, releasing the forward for the next request.
        let answers_json: Vec<(String, String)> =
            answers.iter().map(|(qid, a)| (qid.clone(), answer_to_jev_json(a))).collect();
        Ok(ClassifierResponse {
            model: self.inner.model_name.clone(),
            answers: answers_json,
            usage: Usage { input_tokens: lengths.iter().sum::<usize>() as u64, output_tokens: 0 },
        })
    }
}
