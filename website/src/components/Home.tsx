import { For } from "solid-js";
import { Badge, LinkButton, Mono } from "./ui";
import { CopyBlock } from "./Code";
import { Pre } from "./docs/Prose";
import { highlight } from "../lib/highlight";
import { GITHUB_URL } from "../lib/links";
import { PLATFORMS, LATEST_RELEASE_URL, assetName, downloadUrl } from "../lib/release";

/* ------------------------------------------------------------------ */
/* A fake terminal panel showing a real laya invocation and output.    */
/* ------------------------------------------------------------------ */

function TerminalDemo() {
  const command =
    'laya ask --model-dir laya-typed-decisions \\\n' +
    '  --state "We were billed twice for March. Please refund the duplicate." \\\n' +
    '  --question "Which team should handle this?" \\\n' +
    "  --option billing --option technical --option sales";
  const output = `choice=billing confidence=0.9123 act_p=0.0089
    billing: 0.9123
    technical: 0.0431
    sales: 0.0446`;

  return (
    <div class="overflow-hidden rounded-xl border border-line shadow-2xl">
      <div class="flex items-center gap-2 border-b border-line bg-surface px-4 py-2.5">
        <span class="h-2.5 w-2.5 rounded-full bg-danger" />
        <span class="h-2.5 w-2.5 rounded-full bg-warn" />
        <span class="h-2.5 w-2.5 rounded-full bg-success" />
        <span class="ml-2 font-mono text-xs text-faint">laya</span>
      </div>
      <pre class="overflow-x-auto bg-code-bg px-4 py-4 font-mono text-[0.78rem] leading-relaxed">
        <code class="language-bash">
          <span class="select-none text-faint">$ </span>
          <span innerHTML={highlight(command, "bash")} />
        </code>
        {"\n\n"}
        <code class="language-console">
          <span innerHTML={highlight(output, "console")} />
        </code>
      </pre>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Features.                                                          */
/* ------------------------------------------------------------------ */

const FEATURES = [
  {
    title: "ModernBERT encoder from scratch",
    body: "RoPE with per-layer-type theta, alternating global/sliding-window attention, GeGLU MLP — config-driven, so it runs both the 421M ModernBERT-large and 322M mmBERT-base checkpoints.",
  },
  {
    title: "Decision head",
    body: "Type embedding, 2-layer transformer, option-marker scoring, act (abstention) head — vectorized and autograd-friendly, so the same code path serves both fast inference and training.",
  },
  {
    title: "Typed-question schema builder",
    body: "choice / score / noul rendering and token-budgeted sequence construction, matching the original's [CLS] ... [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP] layout and truncation rules.",
  },
  {
    title: "Calibrated probabilities",
    body: "Per-(question-type, option-count) temperature calibration at inference time, plus a post-hoc temperature-fitting utility.",
  },
  {
    title: "RLCD training",
    body: "REINFORCE with Gaussian-noise exploration and a group-mean baseline (GRPO-style) over a strictly-proper scoring rule, with TD(λ) bootstrapping for multi-turn episodes.",
  },
  {
    title: "Language routing",
    body: "A fast 22-script Unicode detector for the \"wrong alphabet\" case, backed by real statistical language-ID within Latin script — routing never has to guess from model confidence.",
  },
  {
    title: "Fused CUDA kernels",
    body: "RoPE and GeGLU compiled at runtime with NVRTC into single kernels, cutting several elementwise op chains down to one read plus one write each.",
  },
  {
    title: "Unpadded flash-attention",
    body: "--features flash-attn packs every row's real tokens into one flat sequence and runs flash-attn's varlen kernel, so padding is never computed on or attended to.",
  },
  {
    title: "CPU or GPU",
    body: "F32 on CPU by default. --features cuda runs F16 on GPU (matching the original's own default precision and what engages tensor cores); --features flash-attn adds the fused/unpadded path.",
  },
];

function Features() {
  return (
    <div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      <For each={FEATURES}>
        {(f) => (
          <div class="rounded-xl border border-line bg-surface p-5">
            <h3 class="text-[0.9375rem] font-semibold tracking-tight text-ink">{f.title}</h3>
            <p class="mt-2 text-sm leading-relaxed text-muted">{f.body}</p>
          </div>
        )}
      </For>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Install: numbered step cards, the apiplant layout.                  */
/* ------------------------------------------------------------------ */

const stepCard =
  "grid min-w-0 gap-5 rounded-2xl border bg-surface p-5 sm:p-6 lg:grid-cols-[minmax(14rem,0.7fr)_minmax(0,1.3fr)] lg:items-start";

function InstallSteps() {
  const homebrewCommands = `brew tap apiplant/tap
brew install apiplant/tap/laya-rs`;
  const pacmanCommands = `curl -sSfL https://apiplant.github.io/pacman/apiplant.gpg -o /tmp/apiplant.gpg
keyid=$(gpg --show-keys --with-colons /tmp/apiplant.gpg | awk -F: '/^pub:/ { print $5; exit }') && sudo pacman-key --add /tmp/apiplant.gpg && sudo pacman-key --finger "$keyid" && sudo pacman-key --lsign-key "$keyid"
printf '\\n[apiplant]\\nSigLevel = Required DatabaseOptional\\nServer = https://apiplant.github.io/pacman/$arch\\n' | sudo tee -a /etc/pacman.conf > /dev/null
sudo pacman -Sy laya-rs`;
  const aptCommands = `curl -sSfL https://apt.apiplant.com/apiplant-archive-keyring.gpg | sudo tee /usr/share/keyrings/apiplant.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/apiplant.gpg] https://apt.apiplant.com stable main" | sudo tee /etc/apt/sources.list.d/apiplant.list > /dev/null
sudo apt update && sudo apt install laya-rs`;
  const cargoCommands = `cargo add laya-rs         # as a library dependency
cargo install laya-rs     # the laya binary
cargo install laya-rs --features flash-attn  # with CUDA + flash-attn support`;

  return (
    <div class="mt-8 space-y-4 sm:mt-10">
      <div class={`${stepCard} border-accent-line`}>
        <div>
          <div class="flex items-center gap-2">
            <span class="font-mono text-xs text-accent">01</span>
            <Badge tone="accent">Recommended</Badge>
          </div>
          <h3 class="mt-3 text-base font-semibold tracking-tight text-ink">Use a package manager</h3>
          <p class="mt-2 text-sm leading-relaxed text-muted">
            macOS (Apple Silicon), Arch Linux and Debian/Ubuntu are all published to the apiplant
            shared repositories.
          </p>
        </div>

        <div class="min-w-0 space-y-5">
          <div>
            <p class="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-faint">Homebrew</p>
            <CopyBlock command={homebrewCommands} />
          </div>

          <div>
            <p class="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-faint">
              Arch Linux / pacman
            </p>
            <CopyBlock command={pacmanCommands} />
          </div>

          <div>
            <p class="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-faint">
              Debian / Ubuntu
            </p>
            <CopyBlock command={aptCommands} />
          </div>
        </div>
      </div>

      <div class={`${stepCard} border-line`}>
        <div>
          <span class="font-mono text-xs text-accent">02</span>
          <h3 class="mt-3 text-base font-semibold tracking-tight text-ink">Download the archive</h3>
          <p class="mt-2 text-sm leading-relaxed text-muted">
            One archive per platform, holding the <Mono>laya</Mono> binary and the README. No
            installation needed; unpack and run. On Linux x86_64 with an Ampere-or-newer NVIDIA
            GPU, grab the flash-attn archive instead and build/run with{" "}
            <Mono>--features flash-attn</Mono>.
          </p>
        </div>

        <ul class="min-w-0 space-y-1 border-t border-line pt-4 lg:border-t-0 lg:pt-0">
          <For each={PLATFORMS}>
            {(platform) => (
              <li class="min-w-0">
                <a
                  href={downloadUrl(platform)}
                  title={assetName(platform)}
                  class="flex min-w-0 items-baseline justify-between gap-3 rounded-md py-1 text-muted transition-colors hover:text-ink"
                >
                  <span class="shrink-0 text-sm">{platform.label}</span>
                  <span class="min-w-0 truncate font-mono text-xs text-accent">
                    {assetName(platform)}
                  </span>
                </a>
              </li>
            )}
          </For>
        </ul>

        <a
          href={LATEST_RELEASE_URL}
          target="_blank"
          rel="noreferrer noopener"
          class="lg:col-start-2 text-sm font-medium text-accent hover:text-accent-dim"
        >
          All releases and checksums
        </a>
      </div>

      <div class={`${stepCard} border-line`}>
        <div>
          <span class="font-mono text-xs text-faint">03</span>
          <h3 class="mt-3 text-base font-semibold tracking-tight text-ink">Cargo</h3>
          <p class="mt-2 text-sm leading-relaxed text-muted">
            As a library dependency, or to build the CLI from source via crates.io.
          </p>
        </div>

        <div class="min-w-0">
          <CopyBlock command={cargoCommands} />
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Page.                                                              */
/* ------------------------------------------------------------------ */

export function Home() {
  return (
    <div class="mx-auto w-full max-w-6xl px-5">
      {/* Hero */}
      <section class="grid items-center gap-10 py-16 sm:py-20 lg:grid-cols-2 lg:gap-12">
        <div>
          <div class="flex flex-wrap items-center gap-2">
            <Badge tone="accent">v{__VERSION__}</Badge>
            <Badge>Rust</Badge>
            <Badge>candle</Badge>
            <Badge>Apache-2.0</Badge>
          </div>
          <h1 class="mt-5 text-4xl font-semibold tracking-tight text-ink sm:text-5xl">
            Typed decisions, <span class="text-accent">in one forward pass</span>.
          </h1>
          <p class="mt-4 max-w-lg text-lg leading-relaxed text-muted">
            Give it a state — text, email, ticket, or JSON — and typed{" "}
            <Mono>choice</Mono>/<Mono>score</Mono>/<Mono>noul</Mono> questions. Get back
            calibrated probabilities in a single sub-35ms bidirectional pass. No text generation,
            nothing to parse, nothing to hallucinate.
          </p>
          <div class="mt-7">
            <LinkButton
              href="/demo"
              variant="primary"
              class="!px-8 !py-4 !text-lg shadow-lg shadow-accent/20"
            >
              ▶ Try it now — runs in your browser
            </LinkButton>
          </div>
          <div class="mt-4 flex flex-wrap gap-3">
            <LinkButton href={GITHUB_URL} size="lg">
              View on GitHub
            </LinkButton>
            <LinkButton href="/docs" size="lg">
              Read the docs
            </LinkButton>
            <LinkButton href="/#install" size="lg">
              Install
            </LinkButton>
          </div>
          <p class="mt-5 text-sm text-faint">
            A from-scratch Rust reimplementation of{" "}
            <a href="https://laya.convaiinnovations.com/" target="_blank" rel="noreferrer noopener" class="text-accent hover:text-accent-dim">
              Laya
            </a>
            , targeting the published <Mono>convaiinnovations/laya-*</Mono> checkpoint family.
          </p>
        </div>
        <TerminalDemo />
      </section>

      {/* Features */}
      <section id="features" class="pb-16">
        <h2 class="text-2xl font-semibold tracking-tight text-ink">
          A native Rust port, not a wrapper
        </h2>
        <p class="mt-2 max-w-2xl text-muted">
          The encoder, the decision head, the schema builder, and the RLCD training loop are all
          reimplemented from scratch on candle — no PyTorch/Python dependency at inference time.
        </p>
        <div class="mt-8">
          <Features />
        </div>
      </section>

      {/* Install */}
      <section id="install" class="pb-16">
        <h2 class="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">Install</h2>
        <p class="mt-3 max-w-2xl leading-relaxed text-muted">
          Use Homebrew, pacman or apt when your platform has it. Otherwise take the prebuilt
          archive, or pull the library straight from crates.io.
        </p>

        <InstallSteps />
      </section>

      {/* Library */}
      <section class="border-t border-line pb-20 pt-16">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <h2 class="text-2xl font-semibold tracking-tight text-ink">As a library</h2>
          <LinkButton href="/docs/library" size="sm">
            Full library docs →
          </LinkButton>
        </div>
        <p class="mt-3 max-w-2xl leading-relaxed text-muted">
          One <Mono>RLAgent::load</Mono>, one <Mono>system_one</Mono> call. Any number of typed
          questions, scored against one state in a single batched forward pass.
        </p>
        <div class="mt-6">
          <CopyBlock command="cargo add laya-rs" />
        </div>
        <Pre caption="src/main.rs" lang="rust">{`use serde_json::json;
use laya::{QType, Question, RLAgent};

let agent = RLAgent::load("/path/to/laya-typed-decisions")?;
let questions = vec![("urgency".to_string(), Question {
    qtype: QType::Score,
    instructions: "How urgent is this?".to_string(),
    choice_criteria: vec![],
    score_criteria: vec!["not urgent".into(), "soon".into(), "blocking".into()],
    noul_true: None,
    noul_false: None,
})];
let answers = agent.system_one(&json!("We were billed twice for March."), &questions)?;`}</Pre>
      </section>
    </div>
  );
}
