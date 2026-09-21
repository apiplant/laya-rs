import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Pre, FlagTable, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsCli() {
  return (
    <DocsLayout>
      <H1><IC>laya</IC></H1>
      <Lead>
        One binary: a bare-invocation quick demo, plus <IC>ask</IC>, <IC>answer</IC> and{" "}
        <IC>train</IC> subcommands.
      </Lead>

      <Section>
        <H2>Build</H2>
        <CopyBlock command={`cargo build --release                        # CPU\ncargo build --release --features cuda        # CUDA\ncargo build --release --features flash-attn   # CUDA + fused/unpadded flash-attention`} />
      </Section>

      <Section>
        <H2>Bare invocation</H2>
        <P>
          Routes a state string to the right checkpoint and asks three built-in example questions
          — the fastest way to check a checkpoint directory works:
        </P>
        <CopyBlock command={`laya /path/to/laya-typed-decisions \\\n  "We were billed twice for March. Please refund the duplicate."`} />
        <P>
          The first positional argument is the models root (containing <IC>model.safetensors</IC>/
          <IC>tokenizer/</IC> at top level, plus <IC>multilingual/</IC> and{" "}
          <IC>typed-decisions/</IC> subfolders if you keep the whole checkpoint family together);
          the second is the state text.
        </P>
      </Section>

      <Section>
        <H2><IC>laya ask</IC></H2>
        <P>A single <IC>choice</IC> question against a state string — a quick manual smoke test:</P>
        <CopyBlock
          command={`laya ask --model-dir /path/to/laya-typed-decisions \\\n  --state "Alice works for Acme in Paris." \\\n  --question "Which team should handle this?" \\\n  --option billing --option technical --option sales`}
        />
        <Pre caption="stdout" lang="console">{`choice=technical confidence=0.8421 act_p=0.0132
    billing: 0.0524
    technical: 0.8421
    sales: 0.1055`}</Pre>
      </Section>

      <Section>
        <H2><IC>laya answer</IC></H2>
        <P>
          Answers a batch of typed questions against one state (<IC>{"{state, questions}"}</IC>)
          and writes typed answers (<IC>{"{qid: {...answer}}"}</IC>) to an output file:
        </P>
        <CopyBlock command={`laya answer input.json answers.json --model-dir /path/to/laya-typed-decisions`} />
        <P>
          Each question in <IC>questions</IC> is{" "}
          <IC>{"{type: \"choice\"|\"score\"|\"noul\", instructions, criteria}"}</IC>
          {" "}— the same shape a Rust caller builds with <IC>batching::RawQuestion</IC>, and the
          same shape the <a href="/demo" class="text-accent hover:text-accent-dim">browser demo</a>'s
          question builder produces. <IC>--only QID</IC> restricts a run to one question id, useful
          when iterating on a single question's wording.
        </P>
        <Pre caption="input.json" lang="json">{`{
  "state": { "subject": "Duplicate charge on invoice 4411", "body": "..." },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": { "billing": "invoices, payments, refunds", "technical": "bugs and outages" }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["not urgent", "soon", "blocking"]
    },
    "churn_risk": {
      "type": "noul",
      "instructions": "Does the user threaten to cancel?"
    }
  }
}`}</Pre>
        <P>
          <IC>laya answer</IC> only knows one state at a time. To batch-answer a
          jev-questions-style file (<IC>{"{section: {state, questions}}"}</IC>) into{" "}
          <IC>{"{section: {qid: {...answer}}}"}</IC>, use <IC>scripts/jev_batch.py</IC>, which
          calls <IC>laya answer</IC> once per section:
        </P>
        <CopyBlock command={`scripts/jev_batch.py questions.json answers.json --model-dir /path/to/laya-typed-decisions`} />
      </Section>

      <Section>
        <H2>Flags</H2>
        <FlagTable
          rows={[
            { flag: "laya [MODELS_ROOT] [BODY]", meaning: "bare invocation: route + answer three example questions" },
            { flag: "laya ask --model-dir DIR --state S --question Q --option A --option B ...", meaning: "single choice question, printed to stdout" },
            { flag: "laya answer INPUT OUTPUT --model-dir DIR [--only QID]", meaning: "batch-answer typed questions against one state" },
            { flag: "scripts/jev_batch.py INPUT OUTPUT --model-dir DIR [--only QID] [--binary PATH]", meaning: "batch-answer a jev-questions-style file, one `laya answer` call per section" },
            { flag: "laya train MODEL_DIR DATASET [--epochs N] [--lr F] [--group-size N] [--sigma F] [--save-to PATH]", meaning: "RLCD training over a JSONL dataset — see Training" },
          ]}
        />
      </Section>
    </DocsLayout>
  );
}
