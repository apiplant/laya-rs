import { DocsLayout } from "./DocsLayout";
import { H1, H2, H3, Lead, P, UL, LI, IC, Pre, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsLibrary() {
  return (
    <DocsLayout>
      <H1>As a library</H1>
      <Lead>
        Add laya-rs as a Rust dependency to answer typed decisions directly in your process — no
        CLI, no subprocess.
      </Lead>

      <Section>
        <H2>Install</H2>
        <CopyBlock command="cargo add laya-rs" />
        <P>
          Enable GPU inference with <IC>cuda</IC> (or <IC>flash-attn</IC>, which pulls in{" "}
          <IC>cuda</IC> and additionally fuses/unpads the encoder and decision head's attention):{" "}
          <IC>cargo add laya-rs --features flash-attn</IC>.
        </P>
      </Section>

      <Section>
        <H2>Load a checkpoint and ask questions</H2>
        <P>
          <IC>RLAgent::load</IC> reads a checkpoint directory (<IC>rl_agent_config.json</IC>,{" "}
          <IC>tokenizer/</IC>, <IC>encoder/config.json</IC>, <IC>model.safetensors</IC>) and picks
          CUDA + F16 automatically when available, F32 on CPU. <IC>system_one</IC> answers any
          number of typed questions against one <IC>state</IC> (arbitrary JSON, or a plain string)
          in a single batched forward pass:
        </P>
        <Pre caption="src/main.rs" lang="rust">{`use serde_json::json;
use laya::{QType, Question, RLAgent};

let agent = RLAgent::load("/path/to/laya-typed-decisions")?;

let questions = vec![(
    "department".to_string(),
    Question {
        qtype: QType::Choice,
        instructions: "Which team should handle this?".to_string(),
        choice_criteria: vec![
            ("billing".to_string(), Some("invoices, payments, refunds".to_string())),
            ("technical".to_string(), Some("bugs and outages".to_string())),
        ],
        score_criteria: vec![],
        noul_true: None,
        noul_false: None,
    },
)];

let state = json!({ "subject": "Duplicate charge on invoice 4411", "body": "..." });
let answers = agent.system_one(&state, &questions)?;`}</Pre>
      </Section>

      <Section>
        <H2>The three question types</H2>
        <UL>
          <LI>
            <IC>QType::Choice</IC> — <IC>choice_criteria</IC> is an ordered list of{" "}
            <IC>(key, Option&lt;description&gt;)</IC> pairs. Answered as{" "}
            <IC>Answer::Choice {"{ choice, probabilities, confidence, act_probability }"}</IC>: the
            argmax key, every key's probability, and a confidence score derived from the whole
            distribution.
          </LI>
          <LI>
            <IC>QType::Score</IC> — <IC>score_criteria</IC> is an ordered list of ordinal level
            descriptions (order matters; these are never shuffled). Answered as{" "}
            <IC>Answer::Score {"{ score, legend, probabilities, confidence, act_probability }"}</IC>
            : a probability-weighted scalar over level indices, plus the full distribution.
          </LI>
          <LI>
            <IC>QType::Noul</IC> — optional <IC>noul_true</IC>/<IC>noul_false</IC> descriptions
            (falls back to plain "yes"/"no" phrasing). Answered as{" "}
            <IC>Answer::Noul {"{ noul, act_probability }"}</IC>: <IC>noul</IC> is P(true) in{" "}
            <IC>[0, 1]</IC>.
          </LI>
        </UL>
        <P>
          Every answer also carries <IC>act_probability</IC> — the decision head's own abstention
          signal, independent of the option-probability distribution.
        </P>
      </Section>

      <Section>
        <H2>Temperature calibration</H2>
        <P>
          Per-(question-type, option-count) temperatures are read from the checkpoint's{" "}
          <IC>rl_agent_config.json</IC> (<IC>temperature</IC> and{" "}
          <IC>temperature_by_options</IC>, keyed by <IC>batching::temp_bucket</IC>'s cardinality
          buckets — 2, 3-5, 6-10, 11+) and applied to the logits before the softmax that produces{" "}
          <IC>probabilities</IC>, so a 2-option and an 11-option question of the same type are
          calibrated independently. <IC>metrics::confidence_from_probs</IC> turns the resulting
          distribution into the single <IC>confidence</IC> score on every answer.
        </P>
      </Section>

      <Section>
        <H2 id="routing">Language routing</H2>
        <P>
          <IC>router::route(text)</IC> picks between the English and multilingual checkpoints
          deterministically: a dominant non-Latin Unicode script routes to multilingual
          immediately (cheap, and correct even when a model would be confidently wrong — an
          English-only checkpoint scores unreadable scripts with high confidence anyway); Latin
          script or no alphabetic content defers to{" "}
          <a href="https://github.com/pemistahl/lingua-rs" target="_blank" rel="noreferrer noopener" class="text-accent hover:text-accent-dim">
            lingua
          </a>
          , a real statistical language-ID library, to tell English apart from other Latin-script
          languages. Routing is entirely native-only (not compiled into the wasm build powering{" "}
          <a href="/demo" class="text-accent hover:text-accent-dim">the browser demo</a>, which
          lets you pick the checkpoint directly instead).
        </P>
      </Section>

      <Section>
        <H2>Training</H2>
        <P>
          <IC>Trainer</IC> runs the same RLCD (REINFORCE + Gaussian-noise exploration, GRPO-style
          group-mean baseline, TD(λ) for multi-turn episodes) loop the <IC>laya train</IC> CLI
          subcommand wraps. See{" "}
          <a href="/docs/training" class="text-accent hover:text-accent-dim">Training</a> for the
          dataset schema and the reward function.
        </P>
      </Section>

      <Section>
        <H2>CPU, CUDA, or CUDA + flash-attn</H2>
        <P>
          CPU by default (F32). Build with <IC>--features cuda</IC> for GPU inference in F16 —
          matching the original Python implementation's own default precision, and what actually
          engages tensor cores. <IC>--features flash-attn</IC> additionally swaps the encoder's
          and decision head's attention for a fused flash-attention kernel and runs the whole
          stack unpadded (real tokens from every row packed into one flat sequence, row boundaries
          passed to flash-attn's varlen kernel) — laya's fastest path, and the one the top-level
          README's benchmark numbers are measured against.
        </P>
      </Section>

      <Section>
        <H3>Implementation notes</H3>
        <UL>
          <LI><IC>modernbert.rs</IC> — the ModernBERT encoder from scratch: RoPE with per-layer-type theta, alternating global/sliding-window attention, GeGLU MLP, bias-free LayerNorm and rank-3 <IC>Linear</IC> handled explicitly to avoid candle's slow fallback paths.</LI>
          <LI><IC>decision_model.rs</IC> — the from-scratch decision head: type embedding, a 2-layer transformer over option markers, the option-marker scorer, and the act (abstention) head.</LI>
          <LI><IC>schema.rs</IC> — <IC>render_options</IC>/<IC>build_sequence</IC>: the typed-question schema builder, matching the original's <IC>[CLS] ... [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]</IC> layout and truncation rules.</LI>
          <LI><IC>fused.rs</IC> — NVRTC-compiled fused RoPE and GeGLU CUDA kernels (the <IC>cuda</IC> feature; <IC>flash-attn</IC> layers its own fused, unpadded attention on top).</LI>
          <LI><IC>batching.rs</IC> — the record → model-input pipeline shared by training and the <IC>answer</IC> CLI's question format.</LI>
          <LI><IC>train.rs</IC> — the RLCD training loop.</LI>
        </UL>
      </Section>
    </DocsLayout>
  );
}
