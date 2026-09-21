//! wasm-bindgen bindings for running laya in the browser.
//!
//! Model files (`rl_agent_config.json`, `tokenizer/tokenizer.json`,
//! `tokenizer/tokenizer_config.json`, `encoder/config.json`,
//! `model.safetensors`) are fetched by JS (typically from the Hugging Face
//! Hub) and handed to [`WasmAgent::load`] as bytes/strings — there is no
//! filesystem here. Questions and answers go through JSON strings so the JS
//! side never needs a Rust struct layout; the schema is exactly
//! [`crate::batching::RawQuestion`] (`{"type": "choice"|"score"|"noul",
//! "instructions": ..., "criteria": ...}`), the same shape `laya answer`
//! reads from its input file.

use std::collections::BTreeMap;

use serde_json::Value;
use wasm_bindgen::prelude::*;

use crate::agent::answer_to_json;
use crate::batching::{raw_question_to_question, RawQuestion};
use crate::{Question, RLAgent};

fn to_js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Like [`to_js_err`] but keeps an [`anyhow`] error's whole context chain —
/// the outermost context alone ("loading model.safetensors") says nothing
/// about what actually went wrong.
fn anyhow_to_js_err(e: anyhow::Error) -> JsValue {
    JsValue::from_str(&format!("{e:#}"))
}

/// Call once from JS before anything else, to get readable panic messages
/// (from candle shape mismatches etc.) in the browser console.
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct WasmAgent {
    inner: RLAgent,
}

#[wasm_bindgen]
impl WasmAgent {
    /// Loads a checkpoint from its five file contents, on CPU in F32 (the
    /// only combination that makes sense in a browser tab).
    ///
    /// `weights` is `model.safetensors`; `tokenizer` is
    /// `tokenizer/tokenizer.json`; `tokenizer_config_json` is
    /// `tokenizer/tokenizer_config.json`; `encoder_config_json` is
    /// `encoder/config.json`; `rl_agent_config_json` is
    /// `rl_agent_config.json`.
    #[wasm_bindgen]
    pub fn load(
        weights: &[u8],
        tokenizer: &[u8],
        tokenizer_config_json: &str,
        encoder_config_json: &str,
        rl_agent_config_json: &str,
    ) -> Result<WasmAgent, JsValue> {
        let inner = RLAgent::load_from_bytes(
            rl_agent_config_json,
            tokenizer_config_json,
            tokenizer,
            encoder_config_json,
            weights,
        )
        .map_err(anyhow_to_js_err)?;
        Ok(WasmAgent { inner })
    }

    /// Answers a batch of typed questions against one `state` (arbitrary
    /// JSON, or a plain string body).
    ///
    /// `questions_json` is a JSON object of `{qid: {type, instructions,
    /// criteria}}` (see [`crate::batching::RawQuestion`]). Returns JSON
    /// `{qid: {type, ...}}`, one answer per question, in the same shape
    /// `laya answer` writes.
    #[wasm_bindgen]
    pub fn ask(&self, state_json: &str, questions_json: &str) -> Result<String, JsValue> {
        let state: Value = serde_json::from_str(state_json).map_err(to_js_err)?;
        let raw_questions: BTreeMap<String, RawQuestion> = serde_json::from_str(questions_json).map_err(to_js_err)?;
        if raw_questions.is_empty() {
            return Err(to_js_err("no questions given"));
        }
        let questions: Vec<(String, Question)> =
            raw_questions.iter().map(|(qid, rq)| (qid.clone(), raw_question_to_question(rq))).collect();

        let answers = self.inner.system_one(&state, &questions).map_err(anyhow_to_js_err)?;
        let out: BTreeMap<String, Value> = answers.into_iter().map(|(qid, answer)| (qid, answer_to_json(answer))).collect();
        serde_json::to_string(&out).map_err(to_js_err)
    }
}
