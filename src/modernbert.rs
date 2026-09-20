//! Minimal ModernBERT bidirectional encoder (embeddings + alternating global/local RoPE
//! attention + GeGLU MLP), enough to reproduce `transformers.ModernBertModel.forward(...).last_hidden_state`.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::BTreeMap;

thread_local! {
    static PHASE_NS: RefCell<BTreeMap<&'static str, u128>> = RefCell::new(BTreeMap::new());
}

/// Adds `dur` to the running total for `phase`, only when `LAYA_TIMING=1` (checked once by the
/// caller and passed in, so this stays a no-op branch in the hot path otherwise). Synchronizes
/// the device first so the measured duration reflects actual GPU compute, not just dispatch.
fn record_phase(debug_timing: bool, device: &Device, phase: &'static str, t0: std::time::Instant) -> Result<()> {
    if !debug_timing {
        return Ok(());
    }
    device.synchronize()?;
    let ns = t0.elapsed().as_nanos();
    PHASE_NS.with(|m| *m.borrow_mut().entry(phase).or_insert(0) += ns);
    Ok(())
}

fn dump_phases() {
    PHASE_NS.with(|m| {
        let m = m.borrow();
        let total: u128 = m.values().sum();
        eprintln!("[timing] --- per-phase breakdown (summed across all layers) ---");
        for (phase, ns) in m.iter() {
            eprintln!("[timing]   {phase:<16} {:8.2}ms  ({:5.1}%)", *ns as f64 / 1e6, *ns as f64 / total as f64 * 100.0);
        }
        eprintln!("[timing]   {:<16} {:8.2}ms", "TOTAL", total as f64 / 1e6);
    });
    PHASE_NS.with(|m| m.borrow_mut().clear());
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModernBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub norm_eps: f64,
    pub local_attention: usize,
    pub global_attn_every_n_layers: usize,
    pub layer_types: Vec<String>,
    pub rope_parameters: RopeParams,
    pub pad_token_id: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RopeParams {
    pub full_attention: RopeEntry,
    pub sliding_attention: RopeEntry,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RopeEntry {
    pub rope_theta: f64,
}

/// ModernBERT's norms are bias-free, but `LayerNorm::new_no_bias` is a performance trap in
/// candle: the fused CUDA kernel is only taken when a bias is present, so a bias-free norm falls
/// back to an ~8-op path that upcasts the whole tensor to F32. Adding an explicit zero bias is
/// mathematically identical and ~10x faster (measured 0.66ms -> 0.06ms per call at [7,1024,1024]).
fn layer_norm_no_bias(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = Tensor::zeros(size, weight.dtype(), weight.device())?;
    Ok(LayerNorm::new(weight, bias, eps))
}

fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(weight, None))
}

/// `candle_nn::Linear::forward` uses `broadcast_matmul`, which on a rank-3 input issues a
/// *batched* GEMM (one per batch row) instead of collapsing the leading dims into a single large
/// GEMM. For [7,1024,1024] x [1024,5248] that measured 1.21ms vs 0.53ms for the equivalent
/// flattened [7168,1024] matmul — so flatten first, exactly like torch's `F.linear` does.
pub fn linear_flat(l: &Linear, x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    if dims.len() <= 2 {
        return l.forward(x);
    }
    let in_dim = dims[dims.len() - 1];
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let out = l.forward(&x.reshape((rows, in_dim))?)?;
    let out_dim = out.dim(D::Minus1)?;
    let mut shape = dims[..dims.len() - 1].to_vec();
    shape.push(out_dim);
    out.reshape(shape)
}

struct MlpGeglu {
    wi: Linear,
    wo: Linear,
}

impl MlpGeglu {
    fn new(hidden: usize, intermediate: usize, vb: VarBuilder) -> Result<Self> {
        let wi = linear_no_bias(hidden, intermediate * 2, vb.pp("Wi"))?;
        let wo = linear_no_bias(intermediate, hidden, vb.pp("Wo"))?;
        Ok(Self { wi, wo })
    }

    fn forward(&self, x: &Tensor, debug_timing: bool) -> Result<Tensor> {
        let t0 = std::time::Instant::now();
        let x = linear_flat(&self.wi, x)?;
        record_phase(debug_timing, x.device(), "mlp.wi", t0)?;

        let t0 = std::time::Instant::now();
        let gated = if crate::fused::usable(&x) {
            crate::fused::geglu(&x)?
        } else {
            let last = x.dim(D::Minus1)?;
            let half = last / 2;
            let gate = x.narrow(D::Minus1, 0, half)?.gelu_erf()?;
            let up = x.narrow(D::Minus1, half, half)?;
            (gate * up)?
        };
        record_phase(debug_timing, gated.device(), "mlp.gelu_gate", t0)?;

        let t0 = std::time::Instant::now();
        let out = linear_flat(&self.wo, &gated)?;
        record_phase(debug_timing, out.device(), "mlp.wo", t0)?;
        Ok(out)
    }
}

/// Packing info for the variable-length flash-attention path: lets ragged batches (rows with
/// different real/unpadded lengths) use the fused kernel by concatenating only the real tokens
/// into one flat sequence and telling flash-attn where each row starts via `cu_seqlens`. Built
/// once per forward pass, reused by every layer.
pub struct VarLenPack {
    /// Flat indices (into a `[b*s, ...]` view) of every real, non-padding token, row by row.
    pub indices: Tensor,
    /// Cumulative sequence lengths, `b + 1` elements: `[0, len_0, len_0+len_1, ...]` (u32).
    pub cu_seqlens: Tensor,
    pub max_seqlen: usize,
    pub total_tokens: usize,
    /// Per-token position within its own sequence, for RoPE (the packed layout loses the
    /// positional meaning of the flat index).
    pub pos_ids: Tensor,
}

/// Builds the unpadding plan for a batch, or `None` when every row already has the same real
/// length (nothing to skip, so the dense kernel applies directly). Shared by the encoder and the
/// decision head so both agree on the packed layout.
pub fn build_varlen_pack(attention_mask: &Tensor, s: usize, device: &Device) -> Result<Option<VarLenPack>> {
    let lens: Vec<usize> = attention_mask
        .to_dtype(DType::F32)?
        .sum(1)?
        .to_vec1::<f32>()?
        .iter()
        .map(|v| *v as usize)
        .collect();
    if lens.windows(2).all(|w| w[0] == w[1]) {
        return Ok(None);
    }
    let total: usize = lens.iter().sum();
    let mut indices: Vec<u32> = Vec::with_capacity(total);
    let mut pos: Vec<u32> = Vec::with_capacity(total);
    let mut cu: Vec<u32> = Vec::with_capacity(lens.len() + 1);
    cu.push(0);
    let mut acc = 0u32;
    for (row, &len) in lens.iter().enumerate() {
        // Sequences are left-aligned (collate fills ids[..len] then pads), so a row's real
        // tokens are the first `len` slots of its [b*s] span.
        let base = (row * s) as u32;
        indices.extend((0..len as u32).map(|i| base + i));
        pos.extend(0..len as u32);
        acc += len as u32;
        cu.push(acc);
    }
    Ok(Some(VarLenPack {
        indices: Tensor::from_vec(indices, total, device)?,
        cu_seqlens: Tensor::from_vec(cu, lens.len() + 1, device)?,
        max_seqlen: lens.iter().copied().max().unwrap_or(0),
        total_tokens: total,
        pos_ids: Tensor::from_vec(pos, total, device)?,
    }))
}

/// How `Attention::forward` should compute attention for one layer.
pub enum AttnMode<'a> {
    /// Naive 4-op path (matmul, mask add, softmax, matmul); always correct, handles padding.
    Naive,
    /// Fused flash-attention kernel. `window` is `None` for full attention (global layers) or
    /// `Some(local_attention / 2)` for sliding layers. `pack` is `None` when every row already
    /// has the same real length (no padding to skip), otherwise the varlen packing info — either
    /// way padded positions never participate, which is what makes this safe despite flash-attn
    /// having no key-padding-mask input.
    Flash { window: Option<usize>, pack: Option<&'a VarLenPack> },
}

struct Attention {
    wqkv: Linear,
    wo: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(hidden: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        let wqkv = linear_no_bias(hidden, hidden * 3, vb.pp("Wqkv"))?;
        let wo = linear_no_bias(hidden, hidden, vb.pp("Wo"))?;
        Ok(Self { wqkv, wo, n_heads, head_dim: hidden / n_heads })
    }

    #[allow(unused_variables)]
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor, mode: &AttnMode, debug_timing: bool) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;

        let t0 = std::time::Instant::now();
        let qkv = linear_flat(&self.wqkv, x)?; // [b, s, 3*hidden]
        record_phase(debug_timing, x.device(), "attn.wqkv", t0)?;
        let qkv = qkv.reshape((b, s, 3, self.n_heads, self.head_dim))?;

        #[cfg(feature = "flash-attn")]
        if let AttnMode::Flash { window, pack } = mode {
            let t0 = std::time::Instant::now();
            // Stay in [b,s,h,d] the whole way: flash-attn reads per-dimension strides and only
            // requires the *last* dim to be contiguous, so these narrow+squeeze views need no
            // copy at all (the naive path below has to transpose to [b,h,s,d] and materialize
            // three full [b,h,s,d] tensors per layer just to feed `matmul`).
            let q = qkv.narrow(2, 0, 1)?.squeeze(2)?;
            let k = qkv.narrow(2, 1, 1)?.squeeze(2)?;
            let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;
            // cos/sin arrive as [1,1,s,d] for the naive layout; same memory reinterpreted as
            // [1,s,1,d] broadcasts correctly over [b,s,h,d].
            let (q, k) = if crate::fused::usable(&qkv) {
                let cos2 = cos.reshape((s, self.head_dim))?;
                let sin2 = sin.reshape((s, self.head_dim))?;
                (
                    crate::fused::rope(&q.contiguous()?, &cos2, &sin2)?,
                    crate::fused::rope(&k.contiguous()?, &cos2, &sin2)?,
                )
            } else {
                let cos = cos.reshape((1, s, 1, self.head_dim))?;
                let sin = sin.reshape((1, s, 1, self.head_dim))?;
                (apply_rope(&q, &cos, &sin)?, apply_rope(&k, &cos, &sin)?)
            };
            record_phase(debug_timing, x.device(), "attn.split+rope", t0)?;

            let t0 = std::time::Instant::now();
            let scale = 1f32 / (self.head_dim as f32).sqrt();
            let hd = self.n_heads * self.head_dim;
            let out = match pack {
                // Dense: one row per sequence, all the same real length, nothing to skip.
                None => candle_flash_attn::flash_attn_windowed(&q, &k, &v, scale, *window, *window)?,
                // Packed: `x` is already `[1, total_real_tokens, hidden]` — the encoder unpadded
                // once on entry — so the varlen kernel just needs the row boundaries. No gather
                // or scatter here; that was costing ~8ms/row/section when done per layer.
                Some(p) => {
                    let flat = |t: &Tensor| -> Result<Tensor> {
                        t.reshape((p.total_tokens, self.n_heads, self.head_dim))
                    };
                    let out = candle_flash_attn::flash_attn_varlen_windowed(
                        &flat(&q)?, &flat(&k)?, &flat(&v.contiguous()?)?,
                        &p.cu_seqlens, &p.cu_seqlens,
                        p.max_seqlen, p.max_seqlen, scale, *window, *window,
                    )?; // [total, h, d]
                    out.reshape((1, p.total_tokens, self.n_heads, self.head_dim))?
                }
            };
            let out = out.reshape((b, s, hd))?;
            record_phase(debug_timing, x.device(), "attn.flash", t0)?;

            let t0 = std::time::Instant::now();
            let out = linear_flat(&self.wo, &out)?;
            record_phase(debug_timing, x.device(), "attn.wo", t0)?;
            return Ok(out);
        }

        let t0 = std::time::Instant::now();
        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?; // [b,h,s,d]
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
        record_phase(debug_timing, x.device(), "attn.split", t0)?;

        let t0 = std::time::Instant::now();
        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;
        record_phase(debug_timing, x.device(), "attn.rope", t0)?;

        // Scale Q (touches b*h*s*d elements) instead of the post-matmul attention scores (b*h*s*s
        // elements) — s >> d here (1024 vs 64), so this is ~16x fewer elementwise ops for the
        // same result (softmax is scale-invariant to *where* the constant factor is applied).
        let t0 = std::time::Instant::now();
        let q = (q * (1f64 / (self.head_dim as f64).sqrt()))?;
        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?; // [b,h,s,s]
        record_phase(debug_timing, x.device(), "attn.qk_matmul", t0)?;

        let t0 = std::time::Instant::now();
        let attn = attn.broadcast_add(mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        record_phase(debug_timing, x.device(), "attn.mask_softmax", t0)?;

        let t0 = std::time::Instant::now();
        let out = attn.matmul(&v)?; // [b,h,s,d]
        record_phase(debug_timing, x.device(), "attn.av_matmul", t0)?;

        let t0 = std::time::Instant::now();
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, s, self.n_heads * self.head_dim))?;
        record_phase(debug_timing, x.device(), "attn.merge", t0)?;

        let t0 = std::time::Instant::now();
        let out = linear_flat(&self.wo, &out)?;
        record_phase(debug_timing, x.device(), "attn.wo", t0)?;
        Ok(out)
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last = x.dim(D::Minus1)?;
    let half = last / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // x: [b,h,s,d], cos/sin: [1,1,s,d]
    let a = x.broadcast_mul(cos)?;
    let b = rotate_half(x)?.broadcast_mul(sin)?;
    a + b
}

/// Matches HF ModernBert's `ModernBertRotaryEmbedding.forward`: freqs/cos/sin are always
/// computed in F32 (explicitly `torch.autocast(..., enabled=False)` there) and only cast to the
/// model's compute dtype (`compute_dtype`, e.g. F16) at the very end, right before being
/// multiplied against Q/K.
fn rope_cos_sin(theta: f64, head_dim: usize, seq_len: usize, compute_dtype: DType, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1f32 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, half, device)?; // [half]
    let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let positions = Tensor::from_vec(positions, seq_len, device)?; // [s]
    let freqs = positions.reshape((seq_len, 1))?.broadcast_mul(&inv_freq.reshape((1, half))?)?; // [s, half]
    let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [s, head_dim]
    let cos = emb.cos()?.reshape((1, 1, seq_len, head_dim))?.to_dtype(compute_dtype)?;
    let sin = emb.sin()?.reshape((1, 1, seq_len, head_dim))?.to_dtype(compute_dtype)?;
    Ok((cos, sin))
}

struct Layer {
    attn_norm: Option<LayerNorm>,
    attn: Attention,
    mlp_norm: LayerNorm,
    mlp: MlpGeglu,
    is_global: bool,
}

pub struct ModernBert {
    tok_embeddings: Embedding,
    emb_norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
    pub config: ModernBertConfig,
    device: Device,
}

impl ModernBert {
    pub fn load(cfg: ModernBertConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let vb_emb = vb.pp("embeddings");
        let tok_weight = vb_emb.pp("tok_embeddings").get((cfg.vocab_size, hidden), "weight")?;
        let tok_embeddings = Embedding::new(tok_weight, hidden);
        let emb_norm = layer_norm_no_bias(hidden, cfg.norm_eps, vb_emb.pp("norm"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let vb_l = vb.pp("layers").pp(i);
            let is_global = cfg.layer_types[i] == "full_attention";
            let attn_norm = if i == 0 {
                None
            } else {
                Some(layer_norm_no_bias(hidden, cfg.norm_eps, vb_l.pp("attn_norm"))?)
            };
            let attn = Attention::new(hidden, cfg.num_attention_heads, vb_l.pp("attn"))?;
            let mlp_norm = layer_norm_no_bias(hidden, cfg.norm_eps, vb_l.pp("mlp_norm"))?;
            let mlp = MlpGeglu::new(hidden, cfg.intermediate_size, vb_l.pp("mlp"))?;
            layers.push(Layer { attn_norm, attn, mlp_norm, mlp, is_global });
        }
        let final_norm = layer_norm_no_bias(hidden, cfg.norm_eps, vb.pp("final_norm"))?;

        Ok(Self { tok_embeddings, emb_norm, layers, final_norm, config: cfg, device: device.clone() })
    }

    /// input_ids: [b, s] i64, attention_mask: [b, s] i64 (1 = real token, 0 = pad)
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let debug_timing = std::env::var("LAYA_TIMING").is_ok();
        let (b, s) = input_ids.dims2()?;
        let mut h = self.tok_embeddings.forward(input_ids)?;
        h = self.emb_norm.forward(&h)?;
        let compute_dtype = h.dtype();

        let t0 = std::time::Instant::now();
        let head_dim = self.config.hidden_size / self.config.num_attention_heads;
        let (cos_g, sin_g) = rope_cos_sin(self.config.rope_parameters.full_attention.rope_theta, head_dim, s, compute_dtype, &self.device)?;
        let (cos_l, sin_l) = rope_cos_sin(self.config.rope_parameters.sliding_attention.rope_theta, head_dim, s, compute_dtype, &self.device)?;

        // Masks are built in F32 (safe arithmetic for the -1e30-ish sentinel) then cast to the
        // compute dtype right before use, same "compute in F32, cast at the boundary" pattern as
        // the RoPE tables above.
        let pad_mask = build_padding_mask(attention_mask)?.to_dtype(compute_dtype)?; // [b,1,1,s] additive
        let local_window = self.config.local_attention / 2;
        let sliding_mask = build_sliding_mask(s, local_window, &self.device)?.to_dtype(compute_dtype)?; // [1,1,s,s] additive
        let global_mask = pad_mask.broadcast_add(&Tensor::zeros((1, 1, s, s), compute_dtype, &self.device)?)?;
        let local_mask = pad_mask.broadcast_add(&sliding_mask)?;

        // Flash-attention has no key-padding-mask input. Uniform-length batches can use the dense
        // kernel directly; ragged ones get a varlen pack (real tokens only, plus cu_seqlens) so
        // padded positions still never participate. Built once per forward, shared by all layers.
        #[cfg(feature = "flash-attn")]
        let (use_flash, pack) = (true, build_varlen_pack(attention_mask, s, &self.device)?);
        #[cfg(not(feature = "flash-attn"))]
        let (use_flash, pack): (bool, Option<VarLenPack>) = (false, None);

        if debug_timing { self.device.synchronize()?; }
        if debug_timing {
            let lens = attention_mask.to_dtype(DType::F32)?.sum(1)?.to_vec1::<f32>()?;
            let real: f32 = lens.iter().sum();
            let padded = (lens.len() * s) as f32;
            eprintln!(
                "[timing]   rope+mask build (s={s}, flash={use_flash}): {:.2}ms | row lens {:?} = {:.0} real / {:.0} padded ({:.0}% waste)",
                t0.elapsed().as_secs_f64() * 1e3,
                lens.iter().map(|v| *v as usize).collect::<Vec<_>>(),
                real, padded, (1.0 - real / padded) * 100.0
            );
        }

        // Packed (unpadded) mode: gather every real token into one flat sequence *once*, run all
        // layers on `[1, total_real_tokens, hidden]`, and scatter back at the end. Every op in a
        // layer except attention is row-wise, so they're indifferent to the packing — and
        // attention gets the row boundaries via cu_seqlens. This is what HF's ModernBERT does in
        // its flash path, and it drops both the per-layer gather/scatter and the compute spent on
        // padding. RoPE needs each token's position *within its own row*, which the flat index no
        // longer encodes, so the tables are re-indexed by `pos_ids` up front.
        let hidden = self.config.hidden_size;
        let (cos_g, sin_g, cos_l, sin_l) = match pack.as_ref() {
            None => (cos_g, sin_g, cos_l, sin_l),
            Some(p) => {
                let take = |t: &Tensor| -> Result<Tensor> {
                    t.reshape((s, head_dim))?.index_select(&p.pos_ids, 0)?.reshape((1, 1, p.total_tokens, head_dim))
                };
                (take(&cos_g)?, take(&sin_g)?, take(&cos_l)?, take(&sin_l)?)
            }
        };
        if let Some(p) = pack.as_ref() {
            h = h.reshape((b * s, hidden))?.index_select(&p.indices, 0)?.reshape((1, p.total_tokens, hidden))?;
        }

        let t0 = std::time::Instant::now();
        for layer in &self.layers {
            let residual = h.clone();
            let normed = match &layer.attn_norm {
                Some(n) => n.forward(&h)?,
                None => h.clone(),
            };
            let (cos, sin, mask) = if layer.is_global {
                (&cos_g, &sin_g, &global_mask)
            } else {
                (&cos_l, &sin_l, &local_mask)
            };
            let mode = if use_flash {
                AttnMode::Flash {
                    window: if layer.is_global { None } else { Some(local_window) },
                    pack: pack.as_ref(),
                }
            } else {
                AttnMode::Naive
            };
            let attn_out = layer.attn.forward(&normed, cos, sin, mask, &mode, debug_timing)?;
            h = (residual + attn_out)?;

            let residual = h.clone();
            let normed = layer.mlp_norm.forward(&h)?;
            let mlp_out = layer.mlp.forward(&normed, debug_timing)?;
            h = (residual + mlp_out)?;
        }
        if debug_timing { self.device.synchronize()?; }
        if debug_timing { eprintln!("[timing]   {} layers: {:.2}ms", self.layers.len(), t0.elapsed().as_secs_f64() * 1e3); }
        if debug_timing { dump_phases(); }

        // Scatter the packed tokens back into their padded slots. Padding positions stay zero:
        // nothing downstream reads them (option markers always sit inside the real content, and
        // the decision head masks padding out anyway).
        if let Some(p) = pack.as_ref() {
            h = Tensor::zeros((b * s, hidden), h.dtype(), h.device())?
                .index_add(&p.indices, &h.reshape((p.total_tokens, hidden))?, 0)?
                .reshape((b, s, hidden))?;
        }
        self.final_norm.forward(&h)
    }
}

fn build_padding_mask(attention_mask: &Tensor) -> Result<Tensor> {
    // attention_mask: [b,s] -> additive mask [b,1,1,s]: 0 for keep, -inf for pad
    let (b, s) = attention_mask.dims2()?;
    let m = attention_mask.to_dtype(DType::F32)?;
    let inv = ((m * -1f64)? + 1f64)?; // 1 where pad
    let neg_inf = (inv * -6.0e4_f64)?;
    neg_inf.reshape((b, 1, 1, s))
}

fn build_sliding_mask(seq_len: usize, window: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let diff = if i > j { i - j } else { j - i };
            if diff > window {
                data[i * seq_len + j] = -6.0e4;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, seq_len, seq_len), device)
}
