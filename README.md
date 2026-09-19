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
