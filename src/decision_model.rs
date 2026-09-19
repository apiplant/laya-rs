//! The from-scratch decision head on top of the ModernBERT encoder: a 2-layer
//! standard (post-attention pre-norm) transformer, an option-marker scorer, and an act head.
//! Mirrors `rl_common.DecisionModel`.

use candle_core::{DType, Tensor, D};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};

use crate::modernbert::{ModernBert, ModernBertConfig};

fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> candle_core::Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = vb.get(out_dim, "bias")?;
    Ok(Linear::new(weight, Some(bias)))
}

fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = vb.get(size, "bias")?;
    Ok(LayerNorm::new(weight, bias, eps))
}

struct HeadLayer {
    norm1: LayerNorm,
    norm2: LayerNorm,
    in_proj_w: Tensor,
    in_proj_b: Tensor,
    out_proj: Linear,
    linear1: Linear,
    linear2: Linear,
    n_heads: usize,
}

impl HeadLayer {
    fn new(d: usize, n_heads: usize, dim_ff: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let norm1 = layer_norm(d, 1e-5, vb.pp("norm1"))?;
        let norm2 = layer_norm(d, 1e-5, vb.pp("norm2"))?;
        let in_proj_w = vb.get((3 * d, d), "self_attn.in_proj_weight")?;
        let in_proj_b = vb.get(3 * d, "self_attn.in_proj_bias")?;
        let out_proj = linear(d, d, vb.pp("self_attn.out_proj"))?;
        let linear1 = linear(d, dim_ff, vb.pp("linear1"))?;
        let linear2 = linear(dim_ff, d, vb.pp("linear2"))?;
        Ok(Self { norm1, norm2, in_proj_w, in_proj_b, out_proj, linear1, linear2, n_heads })
    }

    /// x: [b,s,d], key_padding_additive: [b,1,1,s] additive mask (0 keep, -inf pad)
    fn forward(&self, x: &Tensor, key_padding_additive: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x.clone();
        let normed = self.norm1.forward(x)?;
        let attn_out = self.self_attn(&normed, key_padding_additive)?;
        let x = (residual + attn_out)?;

        let residual = x.clone();
        let normed = self.norm2.forward(&x)?;
        let ff = self.linear1.forward(&normed)?.relu()?;
        let ff = self.linear2.forward(&ff)?;
        residual + ff
    }

    fn self_attn(&self, x: &Tensor, key_padding_additive: &Tensor) -> candle_core::Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        let head_dim = d / self.n_heads;
        let qkv = x.broadcast_matmul(&self.in_proj_w.t()?)?.broadcast_add(&self.in_proj_b)?; // [b,s,3d]
        let q = qkv.narrow(D::Minus1, 0, d)?;
        let k = qkv.narrow(D::Minus1, d, d)?;
        let v = qkv.narrow(D::Minus1, 2 * d, d)?;
        let reshape = |t: Tensor| -> candle_core::Result<Tensor> {
            t.reshape((b, s, self.n_heads, head_dim))?.transpose(1, 2)?.contiguous()
        };
        let q = reshape(q)?;
        let k = reshape(k)?;
        let v = reshape(v)?;
        let scale = 1f64 / (head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn = attn.broadcast_add(key_padding_additive)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?; // [b,h,s,hd]
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, s, d))?;
        self.out_proj.forward(&out)
    }
}

pub struct DecisionModel {
    pub encoder: ModernBert,
    type_emb: Embedding,
    head_layers: Vec<HeadLayer>,
    scorer_norm: LayerNorm,
    scorer_l1: Linear,
    scorer_l2: Linear,
    act_l1: Linear,
    act_l2: Linear,
    pub temperature_buf: Vec<f32>,
}

impl DecisionModel {
    pub fn load(
        encoder_cfg: ModernBertConfig,
        head_layers_n: usize,
        n_act: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let d = encoder_cfg.hidden_size;
        let n_heads = (d / 64).max(1);
        let encoder = ModernBert::load(encoder_cfg, vb.pp("encoder"), vb.device())?;

        let type_emb_w = vb.pp("type_emb").get((3, d), "weight")?;
        let type_emb = Embedding::new(type_emb_w, d);

        let mut head_layers = Vec::with_capacity(head_layers_n);
        for i in 0..head_layers_n {
            head_layers.push(HeadLayer::new(d, n_heads, 4 * d, vb.pp("head").pp("layers").pp(i))?);
        }

        let scorer_norm = layer_norm(d, 1e-5, vb.pp("scorer").pp(0))?;
        let scorer_l1 = linear(d, d, vb.pp("scorer").pp(1))?;
        let scorer_l2 = linear(d, 1, vb.pp("scorer").pp(3))?;

        let act_l1 = linear(d + 4, 256, vb.pp("act_head").pp(0))?;
        let act_l2 = linear(256, n_act, vb.pp("act_head").pp(2))?;

        let temperature_buf = vb
            .get(3, "temperature")
            .ok()
            .map(|t: Tensor| t.to_vec1::<f32>().unwrap_or_default())
            .unwrap_or_else(|| vec![1.0, 1.0, 1.0]);

        Ok(Self {
            encoder,
            type_emb,
            head_layers,
            scorer_norm,
            scorer_l1,
            scorer_l2,
            act_l1,
            act_l2,
            temperature_buf,
        })
    }

    /// Vectorized, fully differentiable forward pass — mirrors `DecisionModel.forward` in
    /// `rl_common.py` exactly (gather markers, score, mask, act head on pooled + detached feats).
    ///
    /// `marker_pos`: [b, kmax] i64 (clamp to 0 for padding slots), `marker_mask`: [b, kmax] f32
    /// (1.0 = real option, 0.0 = padding). Returns (logits [b, kmax], act_logits [b, n_act]).
    pub fn forward_tensors(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        marker_pos: &Tensor,
        marker_mask: &Tensor,
        qtype: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let h = self.encoder.forward(input_ids, attention_mask)?; // [b,s,d]
        let (b, _s, d) = h.dims3()?;

        let type_e = self.type_emb.forward(qtype)?; // [b,d]
        let h = h.broadcast_add(&type_e.reshape((b, 1, d))?)?;

        let pad_additive = build_padding_additive(attention_mask)?; // [b,1,1,s]
        let mut h = h;
        for layer in &self.head_layers {
            h = layer.forward(&h, &pad_additive)?;
        }

        let kmax = marker_pos.dim(1)?;
        let idx = marker_pos.clamp(0i64, i64::MAX)?.to_dtype(DType::U32)?; // [b,kmax]
        let idx = idx.unsqueeze(2)?.broadcast_as((b, kmax, d))?.contiguous()?;
        let m = h.gather(&idx, 1)?; // [b,kmax,d]

        let scored = self.scorer_norm.forward(&m)?;
        let scored = self.scorer_l1.forward(&scored)?.gelu_erf()?;
        let scored = self.scorer_l2.forward(&scored)?.squeeze(D::Minus1)?; // [b,kmax]

        // masked_fill(~marker_mask, -1e4): logits*mask + (mask-1)*1e4
        let neg = ((marker_mask - 1f64)? * 1e4)?;
        let logits = (scored.broadcast_mul(marker_mask)? + neg)?;

        // act head sees pooled CLS + a *detached* summary of the answer distribution (matches
        // `p = torch.softmax(logits.detach(), -1)` in rl_common.py — no grad through feats).
        let logits_detached_vec = logits.detach().to_vec2::<f32>()?;
        let marker_mask_vec = marker_mask.to_vec2::<f32>()?;
        let mut feats_vec = Vec::with_capacity(b * 4);
        for r in 0..b {
            let k = marker_mask_vec[r].iter().filter(|&&m| m > 0.5).count().max(2);
            let valid = &logits_detached_vec[r][..marker_mask_vec[r].iter().filter(|&&m| m > 0.5).count().max(1)];
            let (top0, gap, ent) = softmax_feats(valid);
            feats_vec.extend_from_slice(&[top0, gap, ent, k as f32 / 255.0]);
        }
        let feats = Tensor::from_vec(feats_vec, (b, 4), h.device())?;
        let pooled = h.narrow(1, 0, 1)?.squeeze(1)?; // [b,d], grad-enabled
        let act_in = Tensor::cat(&[&pooled, &feats], D::Minus1)?;
        let act_h = self.act_l1.forward(&act_in)?.gelu_erf()?;
        let act_logits = self.act_l2.forward(&act_h)?; // [b, n_act]

        Ok((logits, act_logits))
    }

    /// Convenience wrapper for inference call sites: builds padded marker tensors from
    /// `Vec<Vec<usize>>`, runs `forward_tensors`, and materializes plain `Vec<Vec<f32>>` output.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        marker_pos: &[Vec<usize>],
        qtype: &[i64],
    ) -> candle_core::Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let device = input_ids.device();
        let b = marker_pos.len();
        let kmax = marker_pos.iter().map(|v| v.len()).max().unwrap_or(0);
        let mut mpos = vec![0i64; b * kmax];
        let mut mmask = vec![0f32; b * kmax];
        for (r, markers) in marker_pos.iter().enumerate() {
            for (c, &p) in markers.iter().enumerate() {
                mpos[r * kmax + c] = p as i64;
                mmask[r * kmax + c] = 1.0;
            }
        }
        let marker_pos_t = Tensor::from_vec(mpos, (b, kmax), device)?;
        let marker_mask_t = Tensor::from_vec(mmask, (b, kmax), device)?;
        let qtype_t = Tensor::from_vec(qtype.to_vec(), b, device)?;

        let (logits, act_logits) = self.forward_tensors(input_ids, attention_mask, &marker_pos_t, &marker_mask_t, &qtype_t)?;
        let act_probs = candle_nn::ops::softmax_last_dim(&act_logits)?;
        Ok((logits.to_vec2::<f32>()?, act_probs.to_vec2::<f32>()?))
    }
}

fn softmax_feats(logits: &[f32]) -> (f32, f32, f32) {
    if logits.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&v| v / sum).collect();
    let mut sorted = probs.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let top0 = sorted[0];
    let top1 = if sorted.len() > 1 { sorted[1] } else { 0.0 };
    let k = probs.len().max(2) as f32;
    let ent = -probs.iter().map(|&p| p * (p.max(1e-9)).ln()).sum::<f32>() / (k.ln());
    (top0, top0 - top1, ent)
}

fn build_padding_additive(attention_mask: &Tensor) -> candle_core::Result<Tensor> {
    let (b, s) = attention_mask.dims2()?;
    let m = attention_mask.to_dtype(DType::F32)?;
    let inv = ((m * -1f64)? + 1f64)?;
    let neg_inf = (inv * f64::from(f32::MIN / 2.0))?;
    neg_inf.reshape((b, 1, 1, s))
}
