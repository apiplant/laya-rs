//! Laya-native answer mapping + response assembly (JEV-005).
//!
//! `answer_to_jev_json` maps a laya [`laya::Answer`] to the Jev/Simple-Jev
//! v1 protocol answer shape (see `docs/jev-server/JEV-005-response-mapping.md`
//! and `spec/attachments/JEV-005/protocol-reference.md`):
//!
//! - choice: `{type, choice, confidence, probabilities}` — candidate keys in
//!   the request's **insertion** order;
//! - score:  `{type, score, confidence, probabilities "0".."N-1", legend}`;
//! - noul:   `{type, noul}` — the native P(true), no confidence field;
//! - the laya `act_probability` action field is **stripped** from every answer.
//!
//! Deliberately distinct from [`laya::agent::answer_to_json`], which BTreeMap-
//! sorts its objects (the crate's `serde_json` keeps sorted-key maps by
//! design — see `Cargo.toml`) and emits `act_probability`. The Jev protocol
//! wants request order for choice probabilities, so this module serializes
//! objects manually, in the order its `Vec` holds them.

use crate::Answer;

/// Token accounting for a response (protocol: `usage`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Usage {
    /// Sum of the actual per-question sequence lengths (the laya backend
    /// re-tokenizes the context per question, so repeated context is
    /// counted).
    pub input_tokens: u64,
    /// Always 0 — no tokens are sampled.
    pub output_tokens: u64,
}

/// The protocol 200 response body:
/// `{"model": ..., "answers": {qid: answer}, "usage": {input_tokens, output_tokens}}`.
///
/// `answers` is held as an ordered `Vec` so question ids serialize in the
/// request's insertion order.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassifierResponse {
    pub model: String,
    pub answers: Vec<(String, String)>,
    pub usage: Usage,
}

/// Serialize one answer to its protocol shape, in the required key order.
pub fn answer_to_jev_json(answer: &Answer) -> String {
    match answer {
        Answer::Choice { choice, probabilities, .. } => {
            // Protocol: confidence = the winning candidate's probability
            // (the max over the distribution, argmax ties broken by request
            // order). The laya `Answer`'s own `confidence` is an internal
            // entropy-based metric and is NOT the protocol value.
            let confidence = probabilities
                .iter()
                .find(|(k, _)| k == choice)
                .map(|(_, p)| *p)
                .unwrap_or_else(|| probabilities.iter().map(|(_, p)| *p).fold(0.0f32, f32::max));
            let mut out = String::from("{\"type\":\"choice\",\"choice\":");
            out.push_str(&serde_json::to_string(choice).expect("string serializes"));
            out.push_str(",\"confidence\":");
            out.push_str(&fmt_f32(confidence));
            out.push_str(",\"probabilities\":{");
            for (i, (k, p)) in probabilities.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).expect("string serializes"));
                out.push_str(":");
                out.push_str(&fmt_f32(*p));
            }
            out.push_str("}}");
            out
        }
        Answer::Score { score, legend, probabilities, .. } => {
            // Protocol: confidence = the largest criterion probability.
            let confidence = probabilities.iter().copied().fold(0.0f32, f32::max);
            let mut out = String::from("{\"type\":\"score\",\"score\":");
            out.push_str(&fmt_f32(*score));
            out.push_str(",\"confidence\":");
            out.push_str(&fmt_f32(confidence));
            out.push_str(",\"probabilities\":{");
            for (i, p) in probabilities.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(&i.to_string()).expect("string serializes"));
                out.push_str(":");
                out.push_str(&fmt_f32(*p));
            }
            out.push_str("},\"legend\":{");
            for (i, c) in legend.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(&i.to_string()).expect("string serializes"));
                out.push_str(":");
                out.push_str(&serde_json::to_string(c).expect("string serializes"));
            }
            out.push_str("}}");
            out
        }
        Answer::Noul { noul, .. } => format!("{{\"type\":\"noul\",\"noul\":{}}}", fmt_f32(*noul)),
    }
}

impl ClassifierResponse {
    /// The full protocol response body, objects in documented key order.
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\"model\":");
        out.push_str(&serde_json::to_string(&self.model).expect("string serializes"));
        out.push_str(",\"answers\":{");
        for (i, (qid, body)) in self.answers.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&serde_json::to_string(qid).expect("string serializes"));
            out.push(':');
            out.push_str(body);
        }
        out.push_str("},\"usage\":{\"input_tokens\":");
        out.push_str(&self.usage.input_tokens.to_string());
        out.push_str(",\"output_tokens\":");
        out.push_str(&self.usage.output_tokens.to_string());
        out.push_str("}}");
        out
    }
}

/// A laya probability/score/noul value as compact JSON text. The values are
/// always finite by construction (softmax outputs, expected-index sums, and
/// clamped native probabilities), so `to_string()` is always valid JSON.
fn fmt_f32(v: f32) -> String {
    v.to_string()
}
