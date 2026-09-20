# laya-rs

A from-scratch Rust reimplementation of [Laya](https://laya.convaiinnovations.com/), Convai
Innovations' sub-35ms, non-autoregressive "System 1" decision engine: give it a **state**
(text, email, ticket, or JSON) and typed **questions** (`choice` / `score` / `noul`), and it
returns typed answers with calibrated probabilities in a single bidirectional forward pass —
no text generation, nothing to parse, nothing to hallucinate.

This port targets the published `convaiinnovations/laya` checkpoint family (ModernBERT-large /
mmBERT-base backbones + a from-scratch decision head) and reimplements the full stack natively
on [candle](https://github.com/huggingface/candle), with no PyTorch/Python dependency at
inference time:

- **ModernBERT encoder from scratch** — RoPE with per-layer-type theta, alternating
  global/sliding-window attention, GeGLU MLP — config-driven, so it runs both the 421M
  ModernBERT-large and 322M mmBERT-base checkpoints.
- **Decision head** — type embedding, 2-layer transformer, option-marker scoring, act head —
  vectorized and autograd-friendly, so the same code path serves both fast inference and training.
- **Typed-question schema builder** — `choice` / `score` / `noul` rendering and token-budgeted
  sequence construction, matching the original's `[CLS] ... [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]
  state [SEP]` layout and truncation rules.
- **Per-(question-type, option-count) temperature calibration** at inference time, plus a
  post-hoc temperature-fitting utility.
- **RLCD training loop** — REINFORCE with Gaussian-noise exploration and a group-mean baseline
  (GRPO-style) over the same strictly-proper scoring rule (log + spherical + ranked probability
  score) the original uses as its reward, with TD(λ) bootstrapping for multi-turn episodes.
- **Language routing** — a fast 22-script Unicode detector for the "wrong alphabet" case, backed
  by [lingua-rs](https://github.com/pemistahl/lingua-rs) for real language identification within
  Latin script, so routing never has to guess from model confidence (an English-only checkpoint
  can be confidently wrong on scripts it can't read).

Comparison against the reference `jev` API and a `gliner` baseline on a typed-decisions fixture:
https://gist.github.com/framp/82a9973988cc41a8b552cb7850b70259

## Performance

CPU by default. For GPU inference:

```bash
cargo build --release --features cuda
```

This runs the encoder + decision head in F16 (matching the original Python implementation's own
default precision, and what actually engages the GPU's tensor cores — inference stayed
accidentally CPU-only, then F32-on-GPU, through earlier iterations of this port; both were real,
measured regressions, not just theoretical ones). On an RTX 4090 this took a 16-question typed-
decisions fixture from several seconds/question down to ~460ms total (details and methodology in
the gist above).

An optional `flash-attn` feature swaps the encoder's attention for a fused flash-attention
kernel (via [candle-flash-attn](https://github.com/huggingface/candle)) where it's safe to —
flash-attn has no key-padding-mask input, so it's only used when every row in a batch has the
same real (unpadded) length, falling back to the naive path otherwise (this is checked once per
batch, not assumed):

```bash
CUDA_COMPUTE_CAP=<your GPU's compute capability, e.g. 89 for Ada/RTX 40xx> \
  cargo build --release --features flash-attn
```

First build clones and compiles NVIDIA's cutlass headers against flash-attention's CUDA kernels,
which takes several minutes (cached after that). On the same fixture this took total time from
~460ms to ~362ms — a further ~21% on top of the F16 fix, landing at roughly 2.2x the original
Python implementation's own latency on identical hardware (was ~2.8x on F16 alone). `CUDARC_CUDA_VERSION`
is pinned in `.cargo/config.toml` since `cudarc` doesn't yet recognize newer CUDA toolkits
without the override.

## Usage

```bash
# quick demo against a checkpoint directory (see laya.convaiinnovations.com for the weights)
laya /path/to/laya-typed-decisions

# ask a single choice question
laya ask --model-dir /path/to/laya --state "..." --question "..." --option "a" --option "b"

# answer a jev-questions-style batch file
laya jev questions.json answers.json --model-dir /path/to/laya-typed-decisions

# RLCD training over a JSONL dataset
laya train /path/to/laya dataset.jsonl --epochs 3
```

Not affiliated with Convai Innovations; this is an independent reimplementation for the Rust
ecosystem. Model weights are not included in this repository — point the commands above at a
local checkpoint directory downloaded from Hugging Face
(`convaiinnovations/laya`, `-multilingual`, `-typed-decisions`).
