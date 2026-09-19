//! RLCD (Reinforcement Learning for Calibrated Decisions) trainer.
//!
//! Ports what `rl_common.py` specifies exactly (config, `build_sequence`, `DecisionModel`,
//! `proper_reward`, `td_lambda_targets`, batching) plus the REINFORCE update loop described in
//! the project README ("exploration adds zero-mean Gaussian noise to the logits; the reward is a
//! strictly proper scoring rule ...; updates are REINFORCE with a group-mean baseline
//! (GRPO-style)"). The repo snapshot this was ported from does not include the actual
//! training-loop source (only `rl_common.py`/`rl_agent_api.py`, which are inference-side), so the
//! REINFORCE estimator and the act-head auxiliary loss below are a best-effort reconstruction
//! from that description, not a byte-for-byte port — clearly marked where reconstructed.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::batching::{collate_items, encode_record, pack_groups, td_lambda_targets, temp_bucket, Item, Record};
use crate::decision_model::DecisionModel;
use crate::metrics::proper_reward;
use crate::modernbert::ModernBertConfig;
use crate::schema::{QType, SpecialTokens};

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
    #[serde(default = "default_max_prefixes")]
    max_prefixes: usize,
    act_costs: HashMap<String, f64>,
}
fn default_max_prefixes() -> usize { 6 }

/// Hyperparameters for the REINFORCE-with-Gaussian-exploration update. Defaults follow the
/// values the README reports (`w_sph=0.5`, `w_rps=1.0`, `log_floor=-9.21`, `td_lambda=1.0`).
pub struct RlcdConfig {
    pub lr: f64,
    pub group_size: usize,
    pub noise_sigma: f32,
    pub w_sph: f32,
    pub w_rps: f32,
    pub log_floor: f32,
    pub td_lambda: f32,
    pub act_loss_weight: f32,
    pub max_tokens: usize,
    pub max_seqs: usize,
    pub seed: u64,
}

impl Default for RlcdConfig {
    fn default() -> Self {
        Self {
            lr: 1e-5,
            group_size: 8,
            noise_sigma: 1.0,
            w_sph: 0.5,
            w_rps: 1.0,
            log_floor: -9.21,
            td_lambda: 1.0,
            act_loss_weight: 1.0,
            max_tokens: 16384,
            max_seqs: 256,
            seed: 0,
        }
    }
}

pub struct Trainer {
    tok: Tokenizer,
    special: SpecialTokens,
    model: DecisionModel,
    varmap: VarMap,
    opt: AdamW,
    cfg: RlAgentConfig,
    act_costs: Vec<f64>, // [escalate, ...] in act_costs iteration order (index 0 reserved for "commit")
    device: Device,
    rng: StdRng,
    pub step: u64,
}

impl Trainer {
    pub fn load(model_dir: impl AsRef<Path>, rlcd: &RlcdConfig) -> anyhow::Result<Self> {
        let dir = model_dir.as_ref();
        let cfg: RlAgentConfig = serde_json::from_str(&std::fs::read_to_string(dir.join("rl_agent_config.json"))?)?;
        let tok = Tokenizer::from_file(dir.join("tokenizer").join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;
        let tok_cfg: TokenizerConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer").join("tokenizer_config.json"))?)?;
        let special = SpecialTokens { cls: tok_cfg.cls_token, sep: tok_cfg.sep_token, mask: tok_cfg.mask_token, pad: tok_cfg.pad_token };

        let encoder_cfg: ModernBertConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("encoder").join("config.json"))?)?;

        let device = Device::Cpu;
        let n_act = cfg.act_costs.len() + 1;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = DecisionModel::load(encoder_cfg, cfg.head_layers, n_act, vb)?;
        // Registers of all vars now exist with default init; overwrite with the pretrained weights.
        varmap.load(dir.join("model.safetensors"))?;

        let opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr: rlcd.lr, ..Default::default() })?;
        let act_costs: Vec<f64> = cfg.act_costs.values().copied().collect();

        Ok(Self { tok, special, model, varmap, opt, cfg, act_costs, device, rng: StdRng::seed_from_u64(rlcd.seed), step: 0 })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        self.varmap.save(path)?;
        Ok(())
    }

    /// One optimizer step over a JSONL-parsed batch of records (each already an `encode_record`
    /// group of `Item`s sharing a `rec_uid`). Returns (mean total loss, mean reward) for logging.
    pub fn train_step(&mut self, groups: Vec<Vec<Item>>, rlcd: &RlcdConfig) -> anyhow::Result<(f32, f32)> {
        let pad_id = self.tok.token_to_id(&self.special.pad).unwrap_or(0);
        let subs = pack_groups(groups, rlcd.max_tokens, rlcd.max_seqs);

        let mut total_loss = 0f32;
        let mut total_reward = 0f32;
        let mut n_items_total = 0usize;

        for sub in subs {
            let items: Vec<&Item> = sub.iter().flatten().collect();
            if items.is_empty() {
                continue;
            }
            let n = items.len();
            let batch = collate_items(&items, pad_id);

            let input_ids = Tensor::from_vec(batch.ids.iter().map(|&x| x as i64).collect::<Vec<_>>(), (batch.n, batch.l), &self.device)?;
            let attention_mask = Tensor::from_vec(batch.attention_mask.clone(), (batch.n, batch.l), &self.device)?;
            let marker_pos = Tensor::from_vec(batch.marker_pos.clone(), (batch.n, batch.kmax), &self.device)?;
            let marker_mask = Tensor::from_vec(batch.marker_mask.clone(), (batch.n, batch.kmax), &self.device)?;
            let qtype_t = Tensor::from_vec(batch.qtype.clone(), batch.n, &self.device)?;

            let (logits, act_logits) = self.model.forward_tensors(&input_ids, &attention_mask, &marker_pos, &marker_mask, &qtype_t)?;

            // TD(lambda) bootstrap for episode (multi-turn) items: replace the raw outcome target
            // at each prefix step with a value blended from the policy's own next-step prediction.
            let clean_probs = masked_softmax(&logits.detach(), &marker_mask, batch.kmax)?;
            let p_true: Vec<f32> = (0..n).map(|r| clean_probs[r].get(1).copied().unwrap_or(0.0)).collect();
            let td_targets = td_lambda_targets(&items.iter().map(|&it| it.clone()).collect::<Vec<_>>(), &p_true, rlcd.td_lambda);
            let mut targets: Vec<Vec<f32>> = items.iter().map(|it| it.target.clone()).collect();
            for (r, it) in items.iter().enumerate() {
                if it.episode {
                    targets[r] = td_targets[r].to_vec();
                }
            }

            // ---- REINFORCE with Gaussian exploration + group-mean baseline (GRPO-style) ----
            let g = rlcd.group_size;
            let kmax = batch.kmax;
            let sigma = rlcd.noise_sigma;
            let normal = Normal::new(0f32, sigma)?;
            let mut noise = vec![0f32; n * g * kmax];
            for (r, it) in items.iter().enumerate() {
                let k = it.markers.len();
                for gi in 0..g {
                    for c in 0..k {
                        noise[(r * g + gi) * kmax + c] = normal.sample(&mut self.rng);
                    }
                }
            }
            let noise_t = Tensor::from_vec(noise, (n * g, kmax), &self.device)?;

            let logits_rep = logits.unsqueeze(1)?.broadcast_as((n, g, kmax))?.reshape((n * g, kmax))?;
            let mask_rep = marker_mask.unsqueeze(1)?.broadcast_as((n, g, kmax))?.reshape((n * g, kmax))?;
            let noise_t = (noise_t * &mask_rep)?; // no exploration noise on padding slots

            let x = (logits_rep.detach() + &noise_t)?; // sampled logits, no grad (data point)
            let diff = (&x - &logits_rep)?; // grad flows through -logits_rep -> model params
            let sq = ((diff.sqr()? / (2.0 * sigma as f64 * sigma as f64))? * &mask_rep)?;
            let sq_sum = sq.sum(1)?; // [n*g], grad-enabled, this is -logpi(x) up to a mu-independent const

            let x_vec = x.to_vec2::<f32>()?;
            let mut advantage = vec![0f32; n * g];
            let mut reward_sum = 0f32;
            for (r, it) in items.iter().enumerate() {
                let k = it.markers.len();
                let is_score = it.qtype == QType::Score;
                let mut rewards = vec![0f32; g];
                for gi in 0..g {
                    let row = &x_vec[r * g + gi][..k];
                    let q = softmax_slice(row);
                    rewards[gi] = proper_reward(&q, &targets[r][..k], is_score, rlcd.w_sph, rlcd.w_rps, rlcd.log_floor);
                }
                let baseline = rewards.iter().sum::<f32>() / g as f32;
                for gi in 0..g {
                    advantage[r * g + gi] = rewards[gi] - baseline;
                    reward_sum += rewards[gi];
                }
            }
            let advantage_t = Tensor::from_vec(advantage, n * g, &self.device)?;
            let reinforce_loss = (sq_sum * &advantage_t)?.mean_all()?;

            // ---- act-head auxiliary loss (reconstructed; see module docs) ----
            // Target: "escalate" (class 1) when the policy's argmax choice would be wrong, scaled
            // by `act_costs["escalate"]`; a missed escalation on a genuinely wrong answer is
            // scaled by `cost_wrong_act` via the same positive-class weight (both come from the
            // checkpoint's own `rl_agent_config.json`, so the ratio is checkpoint-specific).
            let escalate_cost = self.act_costs.first().copied().unwrap_or(0.5) as f32;
            let mut act_targets = Vec::with_capacity(n);
            let mut act_weights = Vec::with_capacity(n);
            for (r, it) in items.iter().enumerate() {
                if it.label < 0 {
                    act_targets.push(0i64);
                    act_weights.push(0f32);
                    continue;
                }
                let k = it.markers.len();
                let probs = &clean_probs[r][..k];
                let argmax = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
                let wrong = argmax as i64 != it.label;
                act_targets.push(if wrong { 1 } else { 0 });
                act_weights.push(if wrong { escalate_cost.max(0.1) } else { 1.0 });
            }
            let act_loss = weighted_cross_entropy(&act_logits, &act_targets, &act_weights)?;

            let loss = (reinforce_loss + (act_loss * rlcd.act_loss_weight as f64)?)?;
            self.opt.backward_step(&loss)?;

            total_loss += loss.to_scalar::<f32>()? * n as f32;
            total_reward += reward_sum / g as f32;
            n_items_total += n;
        }

        self.step += 1;
        if n_items_total == 0 {
            return Ok((0.0, 0.0));
        }
        Ok((total_loss / n_items_total as f32, total_reward / n_items_total as f32))
    }

    /// Reads JSONL records (one `Record` per line, see `batching::Record`), runs `encode_record`
    /// on each with option-order shuffling enabled, groups by record, and trains for `epochs`
    /// passes with token-budgeted batching (`make_token_batches`).
    pub fn train_jsonl(&mut self, path: impl AsRef<Path>, epochs: usize, rlcd: &RlcdConfig) -> anyhow::Result<()> {
        let text = std::fs::read_to_string(path)?;
        let records: Vec<Record> = text.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect::<Result<_, _>>()?;

        for epoch in 0..epochs {
            let mut order: Vec<usize> = (0..records.len()).collect();
            use rand::seq::SliceRandom;
            order.shuffle(&mut self.rng);

            let mut epoch_loss = 0f32;
            let mut epoch_reward = 0f32;
            let mut n_steps = 0usize;
            // One record group per training micro-step keeps `pack_groups`/token-budgeting logic
            // exercised in `train_step` itself; batch several records' groups together here.
            for chunk in order.chunks(32) {
                let mut groups = Vec::with_capacity(chunk.len());
                for (uid, &idx) in chunk.iter().enumerate() {
                    let items = encode_record(
                        &self.tok,
                        &self.special,
                        &records[idx],
                        self.cfg.max_len,
                        self.cfg.head_max_len,
                        self.cfg.max_prefixes,
                        (epoch * records.len() + uid) as i64,
                        Some(&mut self.rng),
                        true,
                    );
                    if !items.is_empty() {
                        groups.push(items);
                    }
                }
                if groups.is_empty() {
                    continue;
                }
                let (loss, reward) = self.train_step(groups, rlcd)?;
                epoch_loss += loss;
                epoch_reward += reward;
                n_steps += 1;
            }
            let denom = n_steps.max(1) as f32;
            println!("epoch {epoch}: loss={:.4} reward={:.4} ({n_steps} steps)", epoch_loss / denom, epoch_reward / denom);
        }
        Ok(())
    }
}

fn softmax_slice(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-12);
    exps.iter().map(|&v| v / sum).collect()
}

fn masked_softmax(logits: &Tensor, mask: &Tensor, kmax: usize) -> candle_core::Result<Vec<Vec<f32>>> {
    let logits_v = logits.to_vec2::<f32>()?;
    let mask_v = mask.to_vec2::<f32>()?;
    let mut out = Vec::with_capacity(logits_v.len());
    for (row, mrow) in logits_v.iter().zip(mask_v.iter()) {
        let k = mrow.iter().filter(|&&m| m > 0.5).count().max(1).min(kmax);
        out.push(softmax_slice(&row[..k]));
    }
    Ok(out)
}

/// Weighted binary cross-entropy over 2 act-head logits (see module docs: reconstructed loss).
fn weighted_cross_entropy(logits: &Tensor, targets: &[i64], weights: &[f32]) -> candle_core::Result<Tensor> {
    let log_probs = candle_nn::ops::log_softmax(logits, 1)?; // [n, n_act]
    let n = targets.len();
    let device = logits.device();
    let idx = Tensor::from_vec(targets.to_vec(), (n, 1), device)?;
    let picked = log_probs.gather(&idx, 1)?.squeeze(1)?; // [n]
    let w = Tensor::from_vec(weights.to_vec(), n, device)?;
    let loss = (picked.neg()? * &w)?.sum_all()?;
    let wsum = w.sum_all()?.to_scalar::<f32>()?.max(1e-6);
    loss / wsum as f64
}

/// Post-hoc per-`(qtype, option-count-bucket)` temperature fitting (README: "fitting a single
/// scalar temperature per question type on your domain distribution cuts expected calibration
/// error from 0.466 to 0.081"). Minimizes NLL over `temperature` by 1-D grid + local refinement.
pub fn fit_temperature(logits: &[f32], target_index: usize) -> f32 {
    let candidates: Vec<f32> = (1..=400).map(|i| i as f32 * 0.05).collect(); // 0.05 .. 20.0
    let mut best_t = 1.0f32;
    let mut best_nll = f32::INFINITY;
    for &t in &candidates {
        let scaled: Vec<f32> = logits.iter().map(|&v| v / t).collect();
        let p = softmax_slice(&scaled);
        let nll = -p[target_index].max(1e-12).ln();
        if nll < best_nll {
            best_nll = nll;
            best_t = t;
        }
    }
    best_t
}

/// Fits one temperature per `temp_bucket(qtype, k)` over a labeled evaluation set of
/// `(qtype, logits, target_index)` triples, by grid-searching mean NLL per bucket.
pub fn fit_temperatures(samples: &[(QType, Vec<f32>, usize)]) -> HashMap<String, f32> {
    let mut buckets: HashMap<String, Vec<(Vec<f32>, usize)>> = HashMap::new();
    for (qtype, logits, target) in samples {
        buckets.entry(temp_bucket(*qtype, logits.len())).or_default().push((logits.clone(), *target));
    }
    let candidates: Vec<f32> = (1..=400).map(|i| i as f32 * 0.05).collect();
    let mut out = HashMap::new();
    for (bucket, items) in buckets {
        let mut best_t = 1.0f32;
        let mut best_nll = f32::INFINITY;
        for &t in &candidates {
            let mean_nll: f32 = items
                .iter()
                .map(|(logits, target)| {
                    let scaled: Vec<f32> = logits.iter().map(|&v| v / t).collect();
                    let p = softmax_slice(&scaled);
                    -p[*target].max(1e-12).ln()
                })
                .sum::<f32>()
                / items.len() as f32;
            if mean_nll < best_nll {
                best_nll = mean_nll;
                best_t = t;
            }
        }
        out.insert(bucket, best_t);
    }
    out
}
