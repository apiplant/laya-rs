import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Pre, Section } from "./Prose";
import { LinkButton, Mono } from "../ui";
import { CopyBlock } from "../Code";
import { GITHUB_URL } from "../../lib/links";

export function DocsOverview() {
  return (
    <DocsLayout>
      <H1>Documentation</H1>
      <Lead>
        laya-rs is a from-scratch Rust (<Mono>candle</Mono>) reimplementation of{" "}
        <a
          href="https://laya.convaiinnovations.com/"
          target="_blank"
          rel="noreferrer noopener"
          class="text-accent hover:text-accent-dim"
        >
          Laya
        </a>
        , Convai Innovations' sub-35ms, non-autoregressive "System 1" decision engine: give it a{" "}
        <strong class="text-ink font-medium">state</strong> (text, email, ticket, or JSON) and
        typed <strong class="text-ink font-medium">questions</strong> (
        <IC>choice</IC>/<IC>score</IC>/<IC>noul</IC>), and it returns typed answers with calibrated
        probabilities in a single bidirectional forward pass — no text generation, nothing to
        parse, nothing to hallucinate. It ships as a library crate and a <IC>laya</IC> command-line
        binary.
      </Lead>

      <Section>
        <H2>Checkpoints</H2>
        <P>
          Two checkpoint families, both published as <IC>convaiinnovations/laya-*</IC> on Hugging
          Face:
        </P>
        <UL>
          <LI>
            <IC>laya-typed-decisions</IC> — English, a 421M-parameter ModernBERT-large backbone.
            The checkpoint tuned for typed choice/score/noul decisions, and the default the{" "}
            <IC>laya answer</IC> subcommand resolves to.
          </LI>
          <LI>
            <IC>laya-multilingual</IC> — a 322M-parameter mmBERT-base backbone, for state text
            outside English or the Latin alphabet.
          </LI>
        </UL>
        <P>
          <IC>router::route</IC> (used by the bare CLI invocation and the <IC>ask</IC> subcommand)
          picks between them deterministically from the state text's script and, within Latin
          script, real language identification — never from model confidence, since an
          English-only checkpoint can be confidently wrong on a script it can't read at all. See{" "}
          <a href="/docs/library#routing" class="text-accent hover:text-accent-dim">
            routing
          </a>{" "}
          in the library docs.
        </P>
        <P>
          Unlike some model-serving CLIs, <IC>laya</IC> does not auto-download checkpoints: point
          it at a local directory containing <IC>rl_agent_config.json</IC>,{" "}
          <IC>tokenizer/tokenizer.json</IC>, <IC>tokenizer/tokenizer_config.json</IC>,{" "}
          <IC>encoder/config.json</IC>, and <IC>model.safetensors</IC> — download it once with{" "}
          <IC>git</IC> or the Hugging Face CLI:
        </P>
        <Pre caption="shell" lang="bash">{`git clone https://huggingface.co/convaiinnovations/laya-typed-decisions
# or: huggingface-cli download convaiinnovations/laya-typed-decisions --local-dir laya-typed-decisions`}</Pre>
      </Section>

      <Section>
        <H2>The CLI</H2>
        <P>
          One binary, <IC>laya</IC>, with three subcommands plus a bare-invocation quick demo:
        </P>
        <UL>
          <LI>
            <a href="/docs/cli" class="text-accent hover:text-accent-dim"><IC>laya ask</IC></a> —
            a single <IC>choice</IC> question against a state string, for a quick manual smoke
            test.
          </LI>
          <LI>
            <a href="/docs/cli" class="text-accent hover:text-accent-dim"><IC>laya answer</IC></a> —
            answers a batch of typed questions against one state (
            <IC>{"{state, questions}"}</IC>) and writes typed answers to an output JSON file.
            <IC>scripts/jev_batch.py</IC> drives it once per section for a jev-questions-style
            batch file (<IC>{"{section: {state, questions}}"}</IC>).
          </LI>
          <LI>
            <a href="/docs/training" class="text-accent hover:text-accent-dim"><IC>laya train</IC></a>{" "}
            — RLCD training over a JSONL dataset.
          </LI>
        </UL>
        <P>
          For using laya-rs as a Rust dependency instead of (or alongside) the CLI, see{" "}
          <a href="/docs/library" class="text-accent hover:text-accent-dim">As a library</a>.
        </P>
      </Section>

      <Section>
        <H2>Install</H2>
        <P>Build the CLI from the crate:</P>
        <CopyBlock command={`cargo build --release                        # CPU\ncargo build --release --features cuda        # CUDA\ncargo build --release --features flash-attn   # CUDA + fused/unpadded flash-attention`} />
        <P>
          Or install from crates.io, or take a prebuilt archive / package for your platform — see
          the <a href="/#install" class="text-accent hover:text-accent-dim">install section</a> on
          the home page for Homebrew, pacman, apt and direct-download options.
        </P>
      </Section>

      <Section>
        <H2>Source and issues</H2>
        <P>The crate, source, and issue tracker all live on GitHub.</P>
        <div class="mt-4">
          <LinkButton href={GITHUB_URL} variant="primary">
            View on GitHub
          </LinkButton>
        </div>
      </Section>
    </DocsLayout>
  );
}
