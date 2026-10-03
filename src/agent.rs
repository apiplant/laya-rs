//! Port of `rl_agent_api.RLAgent`: loads a checkpoint directory (rl_agent_config.json,
//! tokenizer/, encoder/config.json, model.safetensors) and answers typed questions.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use serde_json::{json, Value};
use tokenizers::Tokenizer;

use crate::batching::temp_bucket;
use crate::decision_model::DecisionModel;
use crate::metrics::confidence_from_probs;
use crate::modernbert::ModernBertConfig;
use crate::schema::{build_sequence, render_options, QType, Question, SpecialTokens};

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default = "default_cls")]
    cls_token: String,
    #[serde(default = "default_sep")]
    sep_token: String,
    #[serde(default = "default_mask")]
    mask_token: String,
    #[serde(default = "default_pad")]
    pad_token: String,
}
fn default_cls() -> String { "[CLS]".to_string() }
fn default_sep() -> String { "[SEP]".to_string() }
fn default_mask() -> String { "[MASK]".to_string() }
fn default_pad() -> String { "[PAD]".to_string() }

#[derive(Debug, Deserialize)]
struct RlAgentConfig {
    head_layers: usize,
    max_len: usize,
    head_max_len: usize,
    act_costs: HashMap<String, f64>,
    #[serde(default)]
    temperature: Vec<f32>,
    #[serde(default)]
    temperature_by_options: HashMap<String, f32>,
}

pub struct RLAgent {
    tok: Tokenizer,
    special: SpecialTokens,
    model: DecisionModel,
    cfg: RlAgentConfig,
    device: Device,
}

pub enum Answer {
    Choice { choice: String, probabilities: Vec<(String, f32)>, confidence: f32, act_probability: f32 },
    Score { score: f32, legend: Vec<String>, probabilities: Vec<f32>, confidence: f32, act_probability: f32 },
    Noul { noul: f32, act_probability: f32 },
}

impl RLAgent {
    pub fn load(model_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = model_dir.as_ref();
        let cfg: RlAgentConfig = serde_json::from_str(&std::fs::read_to_string(dir.join("rl_agent_config.json"))?)?;
        let tok = Tokenizer::from_file(dir.join("tokenizer").join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;
        let tok_cfg: TokenizerConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer").join("tokenizer_config.json"))?)?;
        let special = SpecialTokens {
            cls: tok_cfg.cls_token,
            sep: tok_cfg.sep_token,
            mask: tok_cfg.mask_token,
            pad: tok_cfg.pad_token,
        };

        let encoder_cfg: ModernBertConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("encoder").join("config.json"))?)?;

        let device = Device::cuda_if_available(0)?;
        let n_act = cfg.act_costs.len() + 1;
        // F16 on GPU: the checkpoint's own storage dtype, matches the original's default
        // `torch.autocast(dtype=fp16)`, and is what actually uses the RTX 4090's tensor cores —
        // running everything in F32 measured ~4-5x slower for no accuracy benefit (see
        // decision_model.rs / modernbert.rs for the compute-dtype-aware mask/RoPE handling this
        // requires). F32 on CPU, where candle's F16 kernels aren't the fast path.
        let compute_dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], compute_dtype, &device)?
        };
        let model = DecisionModel::load(encoder_cfg, cfg.head_layers, n_act, vb)?;

        Ok(Self { tok, special, model, cfg, device })
    }

    /// The checkpoint's native sequence limits: `max_len` (full compiled
    /// sequence) and `head_max_len` (CLS..SEP head budget). The Jev server
    /// admission layer (JEV-005) uses `max_len` as one half of its effective
    /// cap, `min(--max-model-len, checkpoint max_len)`.
    pub fn checkpoint_limits(&self) -> (usize, usize) {
        (self.cfg.max_len, self.cfg.head_max_len)
    }

    /// The agent's tokenizer (read-only access for the Jev server's
    /// pre-inference sequence admission, JEV-005).
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// The agent's special tokens (read-only; JEV-005 admission builds
    /// sequences with the same tokenizer + specials `system_one` uses).
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special
    }

    /// Same as [`Self::load`] but from in-memory file contents (used by the wasm bindings —
    /// there is no filesystem in a browser tab). Always CPU/F32: the only combination that
    /// makes sense client-side.
    pub fn load_from_bytes(
        rl_agent_config_json: &str,
        tokenizer_config_json: &str,
        tokenizer_bytes: &[u8],
        encoder_config_json: &str,
        weights: &[u8],
    ) -> anyhow::Result<Self> {
        let cfg: RlAgentConfig = serde_json::from_str(rl_agent_config_json)?;
        let tok = Tokenizer::from_bytes(tokenizer_bytes).map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;
        let tok_cfg: TokenizerConfig = serde_json::from_str(tokenizer_config_json)?;
        let special = SpecialTokens {
            cls: tok_cfg.cls_token,
            sep: tok_cfg.sep_token,
            mask: tok_cfg.mask_token,
            pad: tok_cfg.pad_token,
        };
        let encoder_cfg: ModernBertConfig = serde_json::from_str(encoder_config_json)?;

        let device = Device::Cpu;
        let n_act = cfg.act_costs.len() + 1;
        let compute_dtype = DType::F32;
        let tensors = crate::safetensors32::load_buffer(weights, &device)?;
        let vb = VarBuilder::from_tensors(tensors, compute_dtype, &device);
        let model = DecisionModel::load(encoder_cfg, cfg.head_layers, n_act, vb)?;

        Ok(Self { tok, special, model, cfg, device })
    }

    pub fn system_one(&self, state: &Value, questions: &[(String, Question)]) -> anyhow::Result<Vec<(String, Answer)>> {
        let debug_timing = std::env::var("LAYA_TIMING").is_ok();
        let t_tok = crate::timing::Instant::now();
        let mut all_ids = Vec::with_capacity(questions.len());
        let mut all_markers = Vec::with_capacity(questions.len());
        let mut qtypes = Vec::with_capacity(questions.len());

        for (qid, q) in questions {
            let built = build_sequence(&self.tok, &self.special, state, q, self.cfg.max_len, self.cfg.head_max_len);
            let n_opts = render_options(q).len();
            if built.markers.len() != n_opts {
                anyhow::bail!("question {qid:?}: options do not fit in head_max_len={}", self.cfg.head_max_len);
            }
            qtypes.push(q.qtype.as_index());
            all_markers.push(built.markers);
            all_ids.push(built.ids);
        }

        if debug_timing {
            eprintln!("[timing] tokenize/build_sequence ({} questions): {:.2}ms", questions.len(), t_tok.elapsed().as_secs_f64() * 1e3);
        }
        let pad_id = self.tok.token_to_id(&self.special.pad).unwrap_or(0);
        let max_l = all_ids.iter().map(|v| v.len()).max().unwrap_or(0);
        let b = all_ids.len();
        let mut ids_flat = vec![pad_id; b * max_l];
        let mut att_flat = vec![0i64; b * max_l];
        for (r, ids) in all_ids.iter().enumerate() {
            for (c, &id) in ids.iter().enumerate() {
                ids_flat[r * max_l + c] = id;
            }
            for c in 0..ids.len() {
                att_flat[r * max_l + c] = 1;
            }
        }
        let ids_u32: Vec<u32> = ids_flat;
        let input_ids = Tensor::from_vec(ids_u32.iter().map(|&x| x as i64).collect::<Vec<i64>>(), (b, max_l), &self.device)?;
        let attention_mask = Tensor::from_vec(att_flat, (b, max_l), &self.device)?;

        let (logits, act_probs) = self.model.forward(&input_ids, &attention_mask, &all_markers, &qtypes)?;

        let mut out = Vec::with_capacity(questions.len());
        for (r, (qid, q)) in questions.iter().enumerate() {
            let k = all_markers[r].len();
            let qt_idx = q.qtype.as_index() as usize;
            let temp = self
                .cfg
                .temperature_by_options
                .get(&temp_bucket(q.qtype, k))
                .copied()
                .unwrap_or_else(|| self.cfg.temperature.get(qt_idx).copied().unwrap_or(1.0));
            let z: Vec<f32> = logits[r][..k].iter().map(|&v| v / temp).collect();
            let max = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = z.iter().map(|&v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let p: Vec<f32> = exps.iter().map(|&v| v / sum).collect();
            let confidence = confidence_from_probs(&p);
            let act_probability = act_probs[r].get(0).copied().unwrap_or(0.0);

            let answer = match q.qtype {
                QType::Choice => {
                    let keys: Vec<String> = q.choice_criteria.iter().map(|(k, _)| k.clone()).collect();
                    let argmax = p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
                    Answer::Choice {
                        choice: keys[argmax].clone(),
                        probabilities: keys.into_iter().zip(p.iter().copied()).collect(),
                        confidence,
                        act_probability,
                    }
                }
                QType::Score => Answer::Score {
                    score: p.iter().enumerate().map(|(i, &v)| i as f32 * v).sum(),
                    legend: q.score_criteria.clone(),
                    probabilities: p,
                    confidence,
                    act_probability,
                },
                QType::Noul => Answer::Noul { noul: p[1], act_probability },
            };
            out.push((qid.clone(), answer));
        }
        Ok(out)
    }
}

/// `laya answer`'s and the wasm binding's shared output shape: `{"type": "choice"|"score"|"noul", ...}`.
pub fn answer_to_json(answer: Answer) -> Value {
    match answer {
        Answer::Choice { choice, probabilities, confidence, act_probability } => json!({
            "type": "choice", "choice": choice, "confidence": confidence, "act_probability": act_probability,
            "probabilities": probabilities.into_iter().collect::<BTreeMap<_, _>>(),
        }),
        Answer::Score { score, legend, probabilities, confidence, act_probability } => json!({
            "type": "score", "score": score, "confidence": confidence, "act_probability": act_probability,
            "legend": legend.iter().enumerate().map(|(i, c)| (i.to_string(), c.clone())).collect::<BTreeMap<_, _>>(),
            "probabilities": probabilities.iter().enumerate().map(|(i, p)| (i.to_string(), *p)).collect::<BTreeMap<_, _>>(),
        }),
        Answer::Noul { noul, act_probability } => json!({ "type": "noul", "noul": noul, "act_probability": act_probability }),
    }
}
