import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Pre, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsTraining() {
  return (
    <DocsLayout>
      <H1>Training (RLCD)</H1>
      <Lead>
        <IC>laya train</IC> fine-tunes a loaded checkpoint's decision head over a JSONL dataset of
        typed questions and outcomes, using the same strictly-proper scoring rule (log + spherical
        + ranked probability score) the original Laya uses as its reward.
      </Lead>

      <Section>
        <H2>Run it</H2>
        <CopyBlock command={`laya train /path/to/laya-typed-decisions dataset.jsonl \\\n  --epochs 3 --lr 1e-5 --group-size 8 --sigma 1.0 \\\n  --save-to /path/to/laya-typed-decisions/model.trained.safetensors`} />
        <P>
          The loop is REINFORCE with Gaussian-noise exploration and a group-mean baseline
          (GRPO-style): <IC>--group-size</IC> samples are drawn per record with noise{" "}
          <IC>--sigma</IC>, scored, and averaged into a baseline that each sample's advantage is
          measured against — no separate critic network. Multi-turn episodes additionally
          bootstrap with TD(λ), so an outcome known only at the end of a conversation still trains
          every earlier turn's prediction.
        </P>
      </Section>

      <Section>
        <H2>Dataset schema</H2>
        <P>
          One JSON object per line, either a plain record (a state with one or more labeled
          questions) or a multi-turn episode:
        </P>
        <Pre caption="dataset.jsonl — plain record" lang="json">{`{
  "state": { "subject": "Duplicate charge on invoice 4411", "body": "..." },
  "qs": [
    {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": { "billing": "invoices, payments, refunds", "technical": "bugs and outages" },
      "y": 0
    },
    {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["not urgent", "soon", "blocking"],
      "y": 2
    }
  ]
}`}</Pre>
        <P>
          <IC>y</IC> is the index into the question's option order (<IC>choice_criteria</IC>'s
          insertion order, or the ordinal <IC>score_criteria</IC>/fixed <IC>[false, true]</IC> for{" "}
          <IC>noul</IC>) that the model should be pushed toward. A <IC>soft</IC> array of
          probabilities can be given instead of <IC>y</IC> for a distributional target (e.g.
          distilling from an ensemble or human label distribution). <IC>choice</IC> questions have
          their option order shuffled during training (score and noul options never are — ordinal
          and fixed order, respectively) so the head can't key on position.
        </P>
        <Pre caption="dataset.jsonl — episode" lang="json">{`{
  "kind": "episode",
  "ep": {
    "ctx": { "case_id": "case_7719" },
    "turns": [{ "role": "customer", "text": "..." }, { "role": "agent", "text": "..." }],
    "y": 1.0
  },
  "qs": [{ "type": "noul", "instructions": "Will this customer churn?" }]
}`}</Pre>
        <P>
          An episode's single <IC>noul</IC> question is evaluated at several prefix lengths of{" "}
          <IC>turns</IC> (<IC>batching::episode_prefix_lengths</IC> picks up to a max count,
          spread evenly across the conversation), each prefix becoming its own training item
          targeting the same final outcome <IC>y</IC> — TD(λ) then blends each prefix's target
          with the next prefix's prediction rather than training every step directly on the
          (possibly many turns away) final outcome.
        </P>
      </Section>

      <Section>
        <H2>Flags</H2>
        <UL>
          <LI><IC>--epochs N</IC> (default 1) — passes over the dataset.</LI>
          <LI><IC>--lr F</IC> (default 1e-5) — learning rate.</LI>
          <LI><IC>--group-size N</IC> (default 8) — samples per record for the GRPO-style baseline.</LI>
          <LI><IC>--sigma F</IC> (default 1.0) — Gaussian exploration noise scale.</LI>
          <LI><IC>--save-to PATH</IC> — where to write the trained weights (defaults to <IC>model.trained.safetensors</IC> next to the loaded checkpoint).</LI>
        </UL>
      </Section>

      <Section>
        <H2>As a library</H2>
        <P>
          <IC>Trainer::load</IC> and <IC>Trainer::train_jsonl</IC> are the same calls the CLI
          wraps, for driving training from your own harness (e.g. a custom eval loop between
          epochs):
        </P>
        <Pre caption="src/main.rs" lang="rust">{`use laya::{RlcdConfig, Trainer};

let rlcd = RlcdConfig { lr: 1e-5, group_size: 8, noise_sigma: 1.0, ..Default::default() };
let mut trainer = Trainer::load("/path/to/laya-typed-decisions", &rlcd)?;
trainer.train_jsonl("dataset.jsonl", 3, &rlcd)?;
trainer.save("/path/to/laya-typed-decisions/model.trained.safetensors")?;`}</Pre>
      </Section>
    </DocsLayout>
  );
}
