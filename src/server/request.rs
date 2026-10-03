//! Strict parse + validate of the Jev/Simple-Jev v1 classifier request (JEV-003).
//!
//! `parse_classifier_request` is pure over `&[u8]` — no model, no IO. It
//! yields a typed [`ClassifierRequest`] ready for inference, or a
//! `Vec<RequestError>` that the HTTP layer (JEV-004) folds into the protocol
//! 422 envelope (`{error: {message, type, code, param, details[]}}`).
//!
//! Strictness over the CLI's permissive `batching::raw_question_to_question`:
//! unknown question/options/message fields are rejected, criteria
//! cardinality (2–50) is enforced, the state/messages XOR is a cross-field
//! check, and non-string instructions/criteria are rendered with the
//! protocol's [`canonical`] deterministic JSON (byte-for-byte the
//! simple-jev `common/prompt_builder.py::canonical`).
//!
//! ## Insertion order
//!
//! The protocol requires *document* (insertion) order for question ids and
//! choice criteria keys — choice criteria order renders the option prompt and
//! breaks exact ties. Crate-wide `serde_json` uses a sorted `BTreeMap`, so
//! `Value::Object` iteration is sorted, not document order, and enabling
//! `serde_json/preserve_order` would change object-state prompt text across the
//! whole inference/training pipeline (out of scope). Instead, after the main
//! `Value` parse succeeds, a lenient second pass ([`RequestOrder`]) walks the
//! raw bytes with `serde`'s `MapAccess` to capture document order, and the
//! validated `questions`/`choice_criteria` are re-sorted into it.

use std::collections::BTreeMap;
use std::fmt;

use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde::Deserialize;
use serde_json::Value;

use crate::schema::{QType, Question};

/// One protocol error detail: a dotted field path (`[i]` indices for arrays),
/// a human-readable message, and an error type. The router folds these into
/// the 422 envelope (up to 10 details; the rest are counted in the summary).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestError {
    pub param: String,
    pub message: String,
    pub type_: String,
}

fn err(param: impl Into<String>, type_: impl Into<String>, message: impl Into<String>) -> RequestError {
    RequestError { param: param.into(), type_: type_.into(), message: message.into() }
}

/// The text-only roles the laya backend accepts in `messages`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

/// The exactly-one context: shared state, or text chat history.
#[derive(Debug, Clone, PartialEq)]
pub enum Context {
    State(Value),
    Messages(Vec<ChatMessage>),
}

/// A validated classifier request, ready for inference.
#[derive(Debug, Clone)]
pub struct ClassifierRequest {
    pub model: String,
    pub context: Context,
    /// In request document (insertion) order — preserved by the order probe.
    pub questions: Vec<(String, Question)>,
}

/// A valid protocol "entry": a string, JSON object, JSON array, or null.
/// (Bare numbers and booleans are NOT valid entries; nested JSON inside
/// objects/arrays may contain ordinary scalar values.)
fn is_entry(v: &Value) -> bool {
    matches!(v, Value::String(_) | Value::Object(_) | Value::Array(_) | Value::Null)
}

/// The protocol's `canonical()` — deterministic JSON for structured entries:
/// compact separators, object keys sorted, array order retained, unicode
/// readable (not escaped). Port of simple-jev's `common/prompt_builder.py::canonical`
/// (`json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`).
///
/// Object keys are sorted explicitly: the crate's `serde_json` uses a BTreeMap,
/// so a plain `serde_json::to_string` would already sort — but sorting here is
/// what the protocol's `sort_keys=True` mandates regardless.
pub fn canonical(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => serde_json::to_string(s).expect("string serializes"),
        Value::Array(a) => {
            let inner: Vec<String> = a.iter().map(canonical).collect();
            format!("[{}]", inner.join(","))
        }
        Value::Object(m) => {
            // BTreeMap iteration is already lexicographic — the sort_keys the
            // reference's json.dumps(sort_keys=True) requires.
            let inner: Vec<String> = m
                .iter()
                .map(|(k, v)| format!("{}:{}", serde_json::to_string(k).expect("key serializes"), canonical(v)))
                .collect();
            format!("{{{}}}", inner.join(","))
        }
    }
}

/// Render an entry into prompt text: strings verbatim, structured entries as
/// canonical JSON, null as `"null"` (the reference's `render_entry`).
fn render_entry(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => canonical(other),
    }
}

/// Parse + validate a `/v1/classifier` body against the Jev v1 contract.
///
/// Unknown *top-level* fields are ignored (including completion settings like
/// `stream`/`temperature`); unknown *question/options/message* fields are
/// rejected. All errors are accumulated (the envelope carries up to 10).
pub fn parse_classifier_request(bytes: &[u8]) -> Result<ClassifierRequest, Vec<RequestError>> {
    let root: Value = match serde_json::from_slice(bytes) {
        Ok(v) => v,
        Err(e) => return Err(vec![err("", "invalid_json", format!("invalid JSON: {e}"))]),
    };
    let Some(obj) = root.as_object() else {
        return Err(vec![err("", "type_error", "request body must be a JSON object")]);
    };

    let mut errors: Vec<RequestError> = Vec::new();

    // model: required nonempty string. (Equality against the loaded model is
    // the router's check — this module is model-agnostic.)
    let model = match obj.get("model") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        Some(_) => {
            errors.push(err("model", "type_error", "model must be a nonempty string"));
            String::new()
        }
        None => {
            errors.push(err("model", "missing_field", "model is required"));
            String::new()
        }
    };

    // state: string/object/array; an explicit null means "absent". A bare
    // number or boolean is invalid. Empty string/{}/[] DO count as supplied.
    let state = obj.get("state").filter(|v| !v.is_null());
    if let Some(v) = state {
        if !matches!(v, Value::String(_) | Value::Object(_) | Value::Array(_)) {
            errors.push(err("state", "type_error", "state must be a string, JSON object, or JSON array"));
        }
    }

    // messages: array of text chat messages; an explicit null means "absent".
    let messages_present = obj.get("messages").is_some_and(|v| !v.is_null());
    let messages = if !messages_present {
        Vec::new()
    } else {
        match obj.get("messages").expect("presence checked") {
            Value::Array(arr) => {
                if arr.is_empty() {
                    errors.push(err("messages", "too_short", "messages must contain at least one message"));
                }
                parse_messages(arr, &mut errors)
            }
            _ => {
                errors.push(err("messages", "type_error", "messages must be an array of chat messages"));
                Vec::new()
            }
        }
    };

    // Context XOR: exactly one of state / messages supplied (null = absent).
    if state.is_some() == messages_present {
        errors.push(err("state/messages", "context_conflict", "Provide exactly one of state or messages"));
    }

    // questions: required object, 1–256 nonempty keys. Parsed in sorted (BTree)
    // order here; re-sorted into document order by the order probe below.
    let mut questions: Vec<(String, Question)> = Vec::new();
    match obj.get("questions") {
        Some(Value::Object(m)) if !m.is_empty() => {
            if m.len() > 256 {
                errors.push(err("questions", "too_long", "at most 256 questions are allowed"));
            }
            for (qid, qval) in m.iter() {
                if qid.is_empty() {
                    errors.push(err("questions", "empty_key", "question ids must not be empty"));
                    continue;
                }
                if let Some(q) = parse_question(qid, qval, &mut errors) {
                    questions.push((qid.clone(), q));
                }
            }
        }
        Some(Value::Object(_)) => {
            errors.push(err("questions", "too_short", "at least one question is required"));
        }
        Some(_) => errors.push(err("questions", "type_error", "questions must be an object mapping question ids to question definitions")),
        None => errors.push(err("questions", "missing_field", "questions is required")),
    }

    // options: {raw_logits: bool}. raw_logits=true is rejected — the laya
    // backend has no raw-logit diagnostics; unknown options fields are rejected.
    if let Some(v) = obj.get("options") {
        if v.is_null() {
        } else if let Some(o) = v.as_object() {
            for (k, val) in o.iter() {
                if k == "raw_logits" {
                    if !val.is_boolean() {
                        errors.push(err("options.raw_logits", "type_error", "raw_logits must be a boolean"));
                    } else if val.as_bool() == Some(true) {
                        errors.push(err("options.raw_logits", "unsupported", "raw_logits diagnostics are not supported by the laya backend"));
                    }
                } else {
                    errors.push(err(format!("options.{k}"), "extra_forbidden", "unknown options field"));
                }
            }
        } else {
            errors.push(err("options", "type_error", "options must be a JSON object"));
        }
    }

    // Reserved adapter fields: omitted/null/empty are accepted (no effect);
    // non-empty values are rejected by the laya backend.
    for (key, kind) in [("tools", "an array"), ("mm_processor_kwargs", "an object"), ("media_io_kwargs", "an object of objects")] {
        if let Some(v) = obj.get(key).filter(|v| !v.is_null()) {
            let shaped = match key {
                "tools" => v.is_array(),
                _ => v.is_object(),
            };
            if !shaped {
                errors.push(err(key, "type_error", format!("{key} must be {kind}")));
            } else if !v.as_array().is_some_and(|a| a.is_empty()) && !v.as_object().is_some_and(|o| o.is_empty()) {
                errors.push(err(key, "unsupported", format!("{key} is reserved and must be omitted or empty")));
            }
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // Re-sort into document (insertion) order — the protocol assigns choice
    // labels and breaks ties by criteria order, and echoes question order.
    // The main parse already succeeded, so this lenient probe cannot fail.
    let order = RequestOrder::from_bytes(bytes).expect("order probe of an already-validated request");
    let mut by_id: BTreeMap<&str, &Question> = questions.iter().map(|(id, q)| (id.as_str(), q)).collect();
    let mut ordered: Vec<(String, Question)> = Vec::with_capacity(questions.len());
    for qid in &order.questions_order {
        let Some(slot) = by_id.remove(qid.as_str()) else { continue };
        let mut q = slot.clone();
        if let Some(keys) = order.choice_criteria.get(qid) {
            q.choice_criteria = reorder_choice_criteria(q.choice_criteria, keys);
        }
        ordered.push((qid.clone(), q));
    }
    // Any ids the probe missed (shouldn't happen) are appended in sorted order.
    for (id, q) in questions.iter() {
        if !ordered.iter().any(|(oid, _)| oid == id) {
            ordered.push((id.clone(), q.clone()));
        }
    }

    let context = if let Some(s) = state {
        Context::State(s.clone())
    } else {
        Context::Messages(messages)
    };

    Ok(ClassifierRequest { model, context, questions: ordered })
}

/// Reorder `choice_criteria` into the document key `order`, keeping any keys
/// the probe didn't record (defensive) at the end in their current order.
fn reorder_choice_criteria(criteria: Vec<(String, Option<String>)>, order: &[String]) -> Vec<(String, Option<String>)> {
    let mut used = vec![false; criteria.len()];
    let mut out = Vec::with_capacity(criteria.len());
    for key in order {
        if let Some(pos) = criteria.iter().position(|(k, _)| k == key) {
            if !used[pos] {
                used[pos] = true;
                out.push(criteria[pos].clone());
            }
        }
    }
    for (i, item) in criteria.iter().enumerate() {
        if !used[i] {
            out.push(item.clone());
        }
    }
    out
}

/// Text-only chat validation: each message is exactly `{role, content}` with
/// role ∈ system/developer/user/assistant and content a string.
fn parse_messages(arr: &[Value], errors: &mut Vec<RequestError>) -> Vec<ChatMessage> {
    let mut out = Vec::with_capacity(arr.len());
    for (i, msg) in arr.iter().enumerate() {
        let Some(o) = msg.as_object() else {
            errors.push(err(format!("messages[{i}]"), "type_error", "message must be a JSON object"));
            continue;
        };
        for k in o.keys() {
            if k != "role" && k != "content" {
                errors.push(err(
                    format!("messages[{i}].{k}"),
                    "extra_forbidden",
                    "unknown message field (the laya backend supports text-only messages)",
                ));
            }
        }
        let role = match o.get("role") {
            Some(Value::String(r)) if r == "system" => Role::System,
            Some(Value::String(r)) if r == "developer" => Role::Developer,
            Some(Value::String(r)) if r == "user" => Role::User,
            Some(Value::String(r)) if r == "assistant" => Role::Assistant,
            Some(_) => {
                errors.push(err(
                    format!("messages[{i}].role"),
                    "type_error",
                    "role must be system, developer, user, or assistant (text messages only)",
                ));
                continue;
            }
            None => {
                errors.push(err(format!("messages[{i}].role"), "missing_field", "role is required"));
                continue;
            }
        };
        let content = match o.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(_) => {
                errors.push(err(
                    format!("messages[{i}].content"),
                    "type_error",
                    "content must be a string (the laya backend supports text messages only)",
                ));
                continue;
            }
            None => {
                errors.push(err(format!("messages[{i}].content"), "missing_field", "content is required"));
                continue;
            }
        };
        out.push(ChatMessage { role, content });
    }
    out
}

/// One question entry: the `type` discriminator, strict unknown-field
/// rejection, the per-type `criteria` shape/cardinality, and entry rendering
/// (strings verbatim, structured entries via [`canonical`]).
fn parse_question(qid: &str, qval: &Value, errors: &mut Vec<RequestError>) -> Option<Question> {
    let base = format!("questions.{qid}");
    let Some(o) = qval.as_object() else {
        errors.push(err(&base, "type_error", "question must be a JSON object"));
        return None;
    };

    let qtype = match o.get("type") {
        Some(Value::String(s)) if s == "choice" => QType::Choice,
        Some(Value::String(s)) if s == "score" => QType::Score,
        Some(Value::String(s)) if s == "noul" => QType::Noul,
        Some(_) => {
            errors.push(err(format!("{base}.type"), "type_error", "type must be 'choice', 'score', or 'noul'"));
            return None;
        }
        None => {
            errors.push(err(format!("{base}.type"), "missing_field", "type is required"));
            return None;
        }
    };

    for k in o.keys() {
        if !matches!(k.as_str(), "type" | "instructions" | "criteria") {
            errors.push(err(format!("{base}.{k}"), "extra_forbidden", "unknown question field"));
        }
    }

    let instructions = match o.get("instructions") {
        Some(v) if is_entry(v) => render_entry(v),
        Some(_) => {
            errors.push(err(
                format!("{base}.instructions"),
                "type_error",
                "instructions must be a string, JSON object, JSON array, or null",
            ));
            String::new()
        }
        None => {
            errors.push(err(format!("{base}.instructions"), "missing_field", "instructions is required"));
            String::new()
        }
    };

    let mut q = Question {
        qtype,
        instructions,
        choice_criteria: vec![],
        score_criteria: vec![],
        noul_true: None,
        noul_false: None,
    };

    match qtype {
        QType::Choice => {
            let m = match o.get("criteria") {
                Some(Value::Object(m)) => Some(m),
                Some(_) => {
                    errors.push(err(
                        format!("{base}.criteria"),
                        "type_error",
                        "choice criteria must be an object of candidate ids to optional descriptions",
                    ));
                    None
                }
                None => {
                    errors.push(err(format!("{base}.criteria"), "missing_field", "choice criteria is required"));
                    None
                }
            };
            let m = m?;
            if m.len() < 2 {
                errors.push(err(format!("{base}.criteria"), "too_short", "choice criteria requires 2-50 candidates"));
            } else if m.len() > 50 {
                errors.push(err(format!("{base}.criteria"), "too_long", "choice criteria allows at most 50 candidates"));
            }
            for (k, v) in m.iter() {
                if !is_entry(v) {
                    errors.push(err(
                        format!("{base}.criteria.{k}"),
                        "type_error",
                        "criterion description must be a string, JSON object, JSON array, or null",
                    ));
                }
                let desc = match v {
                    Value::Null => None,
                    s => Some(render_entry(s)),
                };
                q.choice_criteria.push((k.clone(), desc));
            }
        }
        QType::Score => {
            let arr = match o.get("criteria") {
                Some(Value::Array(a)) => Some(a),
                Some(_) => {
                    errors.push(err(
                        format!("{base}.criteria"),
                        "type_error",
                        "score criteria must be an ordered array of 2-50 level descriptions (lowest first)",
                    ));
                    None
                }
                None => {
                    errors.push(err(format!("{base}.criteria"), "missing_field", "score criteria is required"));
                    None
                }
            };
            let arr = arr?;
            if arr.len() < 2 {
                errors.push(err(format!("{base}.criteria"), "too_short", "score criteria requires 2-50 levels"));
            } else if arr.len() > 50 {
                errors.push(err(format!("{base}.criteria"), "too_long", "score criteria allows at most 50 levels"));
            }
            for v in arr.iter() {
                if !is_entry(v) {
                    errors.push(err(
                        format!("{base}.criteria"),
                        "type_error",
                        "level description must be a string, JSON object, JSON array, or null",
                    ));
                }
                // null levels keep their ordinal slot and render as "null"
                // (dropping one would shift the rubric indices).
                q.score_criteria.push(render_entry(v));
            }
        }
        QType::Noul => {
            if let Some(v) = o.get("criteria").filter(|v| !v.is_null()) {
                match v.as_object() {
                    Some(m) => {
                        for (k, desc) in m.iter() {
                            if k != "true" && k != "false" {
                                errors.push(err(
                                    format!("{base}.criteria"),
                                    "extra_forbidden",
                                    "noul criteria accepts only the 'true' and 'false' keys",
                                ));
                                continue;
                            }
                            if !is_entry(desc) {
                                errors.push(err(
                                    format!("{base}.criteria.{k}"),
                                    "type_error",
                                    "criterion description must be a string, JSON object, JSON array, or null",
                                ));
                            }
                            let rendered = match desc {
                                Value::Null => None,
                                s => Some(render_entry(s)),
                            };
                            if k == "true" {
                                q.noul_true = rendered;
                            } else {
                                q.noul_false = rendered;
                            }
                        }
                    }
                    None => errors.push(err(
                        format!("{base}.criteria"),
                        "type_error",
                        "noul criteria must be an object with 'true' and/or 'false' keys",
                    )),
                }
            }
        }
    }

    Some(q)
}

// ---------------------------------------------------------------------------
// Document-order probe
//
// The crate's serde_json uses a sorted BTreeMap, so `Value` iteration loses
// document order. This second, lenient pass walks the raw bytes with serde's
// MapAccess to capture the *insertion* order of the top-level `questions` keys
// and, for each `choice` question, the order of its `criteria` keys. It reads
// only those fields and ignores everything else, so it is faster and more
// permissive than the main parse — which means the main parse can never fail
// while this one succeeds.
// ---------------------------------------------------------------------------

/// The captured document order.
#[derive(Debug, Default)]
struct RequestOrder {
    questions_order: Vec<String>,
    choice_criteria: BTreeMap<String, Vec<String>>,
}

impl RequestOrder {
    fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        let mut de = serde_json::Deserializer::from_slice(bytes);
        // The main parse already succeeded and confirmed the top level is an
        // object, so this lenient walk cannot fail. We only read key order.
        let order = RequestOrder::deserialize(&mut de)?;
        Ok(order)
    }
}

struct RequestOrderVisitor;

impl<'de> Visitor<'de> for RequestOrderVisitor {
    type Value = RequestOrder;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("the classifier request object")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<RequestOrder, A::Error> {
        let mut out = RequestOrder::default();
        while let Some(key) = map.next_key::<String>()? {
            if key == "questions" {
                let q = map.next_value::<QuestionsProbe>()?;
                out.questions_order = q.order;
                out.choice_criteria = q.choice_criteria;
            } else {
                map.next_value::<de::IgnoredAny>()?;
            }
        }
        Ok(out)
    }
}

impl<'de> serde::Deserialize<'de> for RequestOrder {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_map(RequestOrderVisitor)
    }
}

/// The `questions` object: captures question-id order and, per choice question,
/// the criteria key order.
#[derive(Default)]
struct QuestionsProbe {
    order: Vec<String>,
    choice_criteria: BTreeMap<String, Vec<String>>,
}

impl<'de> serde::Deserialize<'de> for QuestionsProbe {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_map(QuestionsProbeVisitor)
    }
}

struct QuestionsProbeVisitor;

impl<'de> Visitor<'de> for QuestionsProbeVisitor {
    type Value = QuestionsProbe;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("the questions object")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<QuestionsProbe, A::Error> {
        let mut out = QuestionsProbe::default();
        while let Some(qid) = map.next_key::<String>()? {
            let q = map.next_value::<QuestionProbe>()?;
            if let Some(keys) = q.choice_criteria_keys {
                out.choice_criteria.insert(qid.clone(), keys);
            }
            out.order.push(qid);
        }
        Ok(out)
    }
}

/// One question object: reads `type` and, if it is a choice, captures the
/// `criteria` key order. Everything else is ignored (the probe is lenient).
#[derive(Default)]
struct QuestionProbe {
    choice_criteria_keys: Option<Vec<String>>,
}

impl<'de> serde::Deserialize<'de> for QuestionProbe {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_map(QuestionProbeVisitor)
    }
}

struct QuestionProbeVisitor;

impl<'de> Visitor<'de> for QuestionProbeVisitor {
    type Value = QuestionProbe;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a question object")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<QuestionProbe, A::Error> {
        let mut qtype: Option<Value> = None;
        let mut crit_keys: Vec<String> = Vec::new();
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "type" => qtype = Some(map.next_value::<Value>()?),
                "criteria" => crit_keys = map.next_value::<KeySeq>()?.0,
                _ => {
                    map.next_value::<de::IgnoredAny>()?;
                }
            }
        }
        let keys = match qtype {
            Some(Value::String(t)) if t == "choice" => Some(crit_keys),
            _ => None,
        };
        Ok(QuestionProbe { choice_criteria_keys: keys })
    }
}

/// Captures the keys of a JSON object *in document order*; for any non-object
/// value (array, scalar, null) it yields an empty vec so it never errors.
#[derive(Debug, PartialEq, Eq)]
struct KeySeq(Vec<String>);

impl<'de> serde::Deserialize<'de> for KeySeq {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(KeySeqVisitor).map(KeySeq)
    }
}

struct KeySeqVisitor;

impl<'de> Visitor<'de> for KeySeqVisitor {
    type Value = Vec<String>;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("an object (whose keys are captured) or any ignorable value")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Vec<String>, A::Error> {
        let mut keys = Vec::new();
        while let Some(k) = map.next_key::<String>()? {
            keys.push(k);
            map.next_value::<de::IgnoredAny>()?;
        }
        Ok(keys)
    }
    // Non-object values carry no key order — yield empty rather than error.
    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> { Ok(vec![]) }
    fn visit_bool<E: de::Error>(self, _: bool) -> Result<Self::Value, E> { Ok(vec![]) }
    fn visit_i64<E: de::Error>(self, _: i64) -> Result<Self::Value, E> { Ok(vec![]) }
    fn visit_u64<E: de::Error>(self, _: u64) -> Result<Self::Value, E> { Ok(vec![]) }
    fn visit_f64<E: de::Error>(self, _: f64) -> Result<Self::Value, E> { Ok(vec![]) }
    fn visit_str<E: de::Error>(self, _: &str) -> Result<Self::Value, E> { Ok(vec![]) }
    fn visit_string<E: de::Error>(self, _: String) -> Result<Self::Value, E> { Ok(vec![]) }
    fn visit_bytes<E: de::Error>(self, _: &[u8]) -> Result<Self::Value, E> { Ok(vec![]) }
    fn visit_byte_buf<E: de::Error>(self, _: Vec<u8>) -> Result<Self::Value, E> { Ok(vec![]) }
    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> { Ok(vec![]) }
    fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<String>, A::Error> {
        while (seq.next_element::<de::IgnoredAny>()?).is_some() {}
        Ok(vec![])
    }
}
