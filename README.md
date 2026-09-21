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

## Installation

macOS (Apple Silicon) and Linux, via Homebrew:

```sh
brew tap apiplant/tap
brew install apiplant/tap/laya-rs
```

Arch Linux, via the signed pacman repository at `apiplant.github.io/pacman`
(one-time setup, then `pacman -Sy`/`-Syu` picks up new releases):

```sh
curl -sSfL https://apiplant.github.io/pacman/apiplant.gpg -o /tmp/apiplant.gpg
keyid=$(gpg --show-keys --with-colons /tmp/apiplant.gpg | awk -F: '/^pub:/ { print $5; exit }') && sudo pacman-key --add /tmp/apiplant.gpg && sudo pacman-key --finger "$keyid" && sudo pacman-key --lsign-key "$keyid"
printf '\n[apiplant]\nSigLevel = Required DatabaseOptional\nServer = https://apiplant.github.io/pacman/$arch\n' | sudo tee -a /etc/pacman.conf > /dev/null
sudo pacman -Sy laya-rs
```

Debian/Ubuntu, via the signed apt repository at `apt.apiplant.com` (one-time
setup, then `apt upgrade` picks up new releases):

```sh
curl -sSfL https://apt.apiplant.com/apiplant-archive-keyring.gpg | sudo tee /usr/share/keyrings/apiplant.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/apiplant.gpg] https://apt.apiplant.com stable main" | sudo tee /etc/apt/sources.list.d/apiplant.list > /dev/null
sudo apt update && sudo apt install laya-rs
```

Or download the archive, `.deb`, or `.pkg.tar.zst` for your platform from the
[releases page](https://github.com/apiplant/laya-rs/releases) and install it
directly — the plain archive needs no installation at all, `laya` is static
enough to run from anywhere. On Linux x86_64 with an Ampere-or-newer NVIDIA
GPU, grab the `laya-rs-flash-attn-*-x86_64-unknown-linux-gnu.tar.gz` archive
instead for CUDA + flash-attention-accelerated inference (needs a host driver
compatible with the CUDA toolkit it was built against).

As a Rust library, or to build the CLI from source, via crates.io:

```sh
cargo add laya-rs         # as a library dependency
cargo install laya-rs     # for the laya binary
cargo install laya-rs --features flash-attn  # with CUDA + flash-attn support
```

| Platform | Ships as |
| --- | --- |
| macOS (Apple Silicon) | archive, Homebrew |
| Linux x86_64 | archive, `.deb` + apt repo, Arch package + pacman repo, Homebrew |
| Linux x86_64, CUDA + flash-attn | archive, `.deb` + apt repo, Arch package + pacman repo |
| Linux aarch64 | archive, `.deb` + apt repo, Homebrew |

No macOS Intel build: only Apple Silicon (`aarch64-apple-darwin`) and Linux
(`x86_64`/`aarch64`) are supported.

See [`packaging/README.md`](packaging/README.md) for how these packages are
built and published.

## Performance

CPU by default. For GPU inference:

```bash
cargo build --release --features cuda
```

This runs in F16 (matching the original Python implementation's own default precision, and what
actually engages the GPU's tensor cores).

An optional `flash-attn` feature additionally swaps the encoder's and decision head's attention
for a fused flash-attention kernel, and runs the whole stack *unpadded* — real tokens from every
row are packed into one flat sequence, with row boundaries passed to flash-attn's varlen kernel,
so padding is never computed on or attended to:

```bash
CUDA_COMPUTE_CAP=<your GPU's compute capability, e.g. 89 for Ada/RTX 40xx> \
  cargo build --release --features flash-attn
```

The first build compiles NVIDIA's cutlass headers against flash-attention's CUDA kernels (a few
minutes, cached afterwards). `CUDARC_CUDA_VERSION` is pinned in `.cargo/config.toml` since
`cudarc` doesn't yet recognize newer CUDA toolkits without the override.

On an RTX 4090, answering a 16-question typed-decisions fixture (details and methodology in the
gist above):

| build | fixture total |
|---|---|
| `--features cuda` | 349 ms |
| `--features flash-attn` | **135 ms** |
| *reference: the original Python implementation, same GPU* | *165 ms* |
| *reference: the jev API this is benchmarked against* | *441 ms* |

On the isolated forward pass at an identical `b=7, s=1024` batch, this port is 60.3 ms against
PyTorch's 63.1 ms (`examples/bench_fwd.rs` mirrors a PyTorch script for a like-for-like number).

Getting there was mostly about finding places where candle silently takes a slow path, which
`examples/bench_ops.rs` (per-op micro-benchmarks at the real layer shapes) and
`examples/bench_fwd.rs` exist to surface:

- `LayerNorm` only uses its fused CUDA kernel when a bias is present. ModernBERT's norms are
  bias-free, so they were taking an ~8-op fallback that upcasts to F32 — 18x slower than the
  memory traffic justifies (37.1 ms → 3.7 ms). Passing an explicit zero bias fixes it.
- `Linear` on a rank-3 input issues a *batched* GEMM instead of one large flattened GEMM (2.3x
  on the model's biggest matmul).
- RoPE and GeGLU were chains of 6 and 3 separate elementwise ops, each a full round trip through
  a 14.7 MB tensor, running several times over their bandwidth bound. `src/fused.rs` replaces
  each with one NVRTC-compiled CUDA kernel (22.8 ms → 2.3 ms and 15.5 ms → 4.4 ms), checked
  against the op chains they replace by unit tests.
- Scaling Q before the QK^T matmul rather than scaling the S×S scores after (~16x fewer
  elementwise ops, since `head_dim` << `seq_len`).
- Unpadding once for the whole encoder instead of gathering/scattering per layer.

Note that per-op timing via `LAYA_TIMING=1` inserts a `device.synchronize()` after each op, which
serializes otherwise-pipelined kernel launches and inflates what it measures — it's useful for
spotting *relative* outliers, but isolated benchmarks and end-to-end wall time are what the
numbers above are based on.

## Usage

```bash
# quick demo against a checkpoint directory (see laya.convaiinnovations.com for the weights)
laya /path/to/laya-typed-decisions

# ask a single choice question
laya ask --model-dir /path/to/laya --state "..." --question "..." --option "a" --option "b"

# answer a batch of typed questions against one state
laya answer input.json answers.json --model-dir /path/to/laya-typed-decisions

# answer a jev-questions-style batch file ({section: {state, questions}}) —
# scripts/jev_batch.py drives `laya answer` once per section
scripts/jev_batch.py questions.json answers.json --model-dir /path/to/laya-typed-decisions

# RLCD training over a JSONL dataset
laya train /path/to/laya dataset.jsonl --epochs 3
```

Not affiliated with Convai Innovations; this is an independent reimplementation for the Rust
ecosystem. Model weights are not included in this repository — point the commands above at a
local checkpoint directory downloaded from Hugging Face
(`convaiinnovations/laya`, `-multilingual`, `-typed-decisions`).
