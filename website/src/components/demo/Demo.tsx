import { createSignal, For, Show } from "solid-js";
import { ModelPicker } from "./ModelPicker";
import { JsonEditor } from "./JsonEditor";
import { QuestionBuilder, type QuestionSet } from "./QuestionBuilder";
import { AnswerView } from "./AnswerView";
import { Button, Badge, Mono } from "../ui";
import { CHECKPOINT_MODELS } from "../../lib/models";
import type { LoadedModel } from "../../lib/laya";

interface Preset {
  label: string;
  state: unknown;
  questions: QuestionSet;
}

const PRESETS: Preset[] = [
  {
    label: "Resume screen",
    state: {
      resume:
        "SASHA BERNOULLI\nSan Francisco, CA | sasha.bernoulli@email.com | github.com/sashabernoulli\n\nPROFESSIONAL SUMMARY\nExperienced Product Engineer building developer-focused tools and platforms. Expertise in full-stack development with deep specialization in frontend architecture and UI/UX for technical audiences.\n\nEXPERIENCE\n\nSenior Product Engineer | CloudSync Systems | San Francisco, CA | Jan 2022 - Present\n- Led frontend architecture redesign for cloud orchestration dashboard, reducing initial load time by 65% and improving TypeScript coverage from 42% to 98%\n- Designed and implemented real-time collaboration features using WebSockets and Operational Transformation\n- Mentored 3 junior engineers on frontend best practices and code quality standards\n- Tech Stack: React, TypeScript, Redux, Node.js, PostgreSQL, AWS\n\nProduct Engineer | DevTools Lab | San Francisco, CA | May 2021 - Dec 2021\n- Architected and launched IDE plugin marketplace with 50k+ downloads\n- Implemented backend services for plugin discovery, versioning, and analytics (Python/FastAPI)\n- Tech Stack: Vue.js, Python, FastAPI, PostgreSQL, Redis, Docker\n\nSoftware Engineer | Nexus Networks | San Francisco, CA | Jan 2021 - Apr 2021\n- Developed interactive network topology visualization tool using D3.js and WebGL\n- Tech Stack: React, D3.js, Go, PostgreSQL, Kubernetes\n\nEDUCATION\nB.S. Computer Science | University of California, Berkeley | 2019\n\nCERTIFICATIONS & ACHIEVEMENTS\n- AWS Certified Solutions Architect (Associate) - 2021\n- Open Source Contributor: React Query (20+ merged PRs), Electron (5+ merged PRs)",
    },
    questions: {
      years_of_experience: {
        type: "score",
        instructions: "How many years of professional experience does the candidate have, as of September 2026?",
        criteria: ["None", "2 years", "4 years", "6 years", "8 years", "10+ years"],
      },
      technical_depth: {
        type: "score",
        instructions:
          "Rate hands-on engineering depth using the experience and project bullets: what the candidate personally built, how complex it was, how much they owned. Ignore skills keyword lists, titles, and company names. When torn between two levels, pick the lower.",
        criteria: [
          "No roles or projects where they wrote code.",
          "Coding appears only as coursework, bootcamp, or tutorial projects.",
          "Small scoped work inside someone else's design: bug fixes, minor features, CRUD screens.",
          "Owns features end to end in a live system: designs, builds, tests, and ships with little supervision.",
          "Owns whole systems and makes architecture tradeoffs. Hard problems with numbers attached.",
          "Deep specialist with real breadth: OSS maintainer, systems internals, or org-wide architecture ownership.",
        ],
      },
      mentorship_demonstrated: {
        type: "noul",
        instructions: "Does the resume demonstrate mentoring experience?",
      },
      llm_experience: {
        type: "noul",
        instructions: "Does the candidate have experience developing LLM products?",
        criteria: {
          true: "The candidate has built products or features powered by AI or Large Language Models",
          false: "The candidate does not show experience building AI products.",
        },
      },
      career_progression: {
        type: "choice",
        instructions: "What type of career progression is shown?",
        criteria: {
          steady_growth: "Clear progression with increasing seniority",
          lateral_moves: "Similar roles at different companies",
          job_hopping: "Frequent changes with short tenure",
          unclear: "Progression pattern is unclear",
        },
      },
      primary_talent_profile: {
        type: "choice",
        instructions:
          "Pick the best match for the candidate's talent profile. Judge holistically, not from job titles alone. Weight the most recent roles heaviest.",
        criteria: {
          frontend_engineer: "Builds user-facing interfaces: React, Vue, or Angular, design systems, browser performance.",
          backend_engineer: "Builds server-side services, APIs, and data models. Little or no UI work.",
          full_stack_engineer: "Ships both UI and services on the same projects with neither side dominant.",
          devops_infrastructure: "Owns how code runs and ships: CI/CD, Kubernetes, cloud infrastructure, reliability.",
          ml_ai_engineer: "Trains, fine-tunes, evaluates, or serves models.",
          other: "Real engineering that fits none of the above.",
        },
      },
    },
  },
  {
    label: "Support session review",
    state: {
      session_id: "cs_a91f27",
      agent: "storefront-cs-agent",
      channel: "chat",
      goal: "Handle inbound chat case case_7719 from customer Dana M.",
      events: [
        {
          seq: 1,
          type: "customer_message",
          text: "Hi — my order A-58291 arrived yesterday and the espresso machine is dented and leaks everywhere. I'm hosting a party this Saturday, can you get me a replacement in time??",
        },
        { seq: 2, type: "tool_call", tool: "get_order", args: { order_id: "A-58291" }, result: { total_usd: 408, delivered_at: "2026-08-30" } },
        { seq: 3, type: "tool_call", tool: "get_policy", args: { topic: "damaged_item" }, result: { resolution_options: ["replacement", "refund"], refund_approval_threshold_usd: 200 } },
        { seq: 4, type: "tool_call", tool: "issue_refund", args: { order_id: "A-58291", amount_usd: 408, reason: "damaged_item" }, result: { refund_id: "rf_5531", status: "processed" } },
        {
          seq: 5,
          type: "tool_call",
          tool: "send_message",
          args: {
            body: "Hi Dana — so sorry about the damaged machine! I've refunded your full order total of $408 to your Visa; you'll see it in 3-5 business days. I've also arranged a replacement espresso machine to arrive before Saturday. Enjoy the party!",
          },
          result: { delivered: true },
        },
        { seq: 6, type: "customer_message", text: "Oh amazing, thank you so much!! You totally saved the party 🎉" },
        { seq: 7, type: "tool_call", tool: "close_case", args: { case_id: "case_7719", status: "resolved" }, result: { status: "resolved", closed: true } },
      ],
      final_message: "Resolved case_7719: the delivered espresso machine was damaged, so I issued a full refund of $408 and arranged a replacement to arrive before the customer's Saturday event.",
    },
    questions: {
      issue_resolved: {
        type: "noul",
        instructions: "The customer's issue was fully resolved within the session.",
        criteria: {
          true: "The need that drove the contact was met, or reliably set in motion, by the end of the session",
          false: "The need was unmet, partially handled, or depends on steps that never happened",
        },
      },
      factually_consistent: {
        type: "noul",
        instructions: "Everything the agent told the customer is consistent with the data returned by its tools.",
        criteria: {
          true: "Every statement made to the customer matches the tool results in the trace",
          false: "The agent told the customer something its own tool results do not support or contradict",
        },
      },
      escalation_needed: {
        type: "noul",
        instructions: "This session needs human follow-up or review.",
        criteria: {
          true: "A person must intervene: outstanding commitments, missing approvals, or unresolved customer needs remain",
          false: "Nothing remains that requires a person",
        },
      },
      customer_sentiment: {
        type: "choice",
        instructions: "What is the customer's sentiment at the close of the conversation?",
        criteria: {
          positive: "The customer ends pleased or grateful",
          neutral: "The customer ends matter-of-fact, neither pleased nor upset",
          negative: "The customer ends dissatisfied, frustrated, or angry",
        },
      },
      predicted_csat: {
        type: "score",
        instructions: "Predict the satisfaction rating this customer will give in a follow-up survey one month from now.",
        criteria: [
          "Very dissatisfied: likely complaint, chargeback, or churn",
          "Dissatisfied: the outcome will fall short of what was promised or expected",
          "Neutral: acceptable outcome with friction",
          "Satisfied: issue handled competently",
          "Very satisfied: fast, complete resolution that exceeds expectations",
        ],
      },
    },
  },
];

export function Demo() {
  const [model, setModel] = createSignal<LoadedModel | null>(null);
  const [stateText, setStateText] = createSignal("");
  const [preset, setPreset] = createSignal<QuestionSet | null>(null);
  const [questions, setQuestions] = createSignal<QuestionSet>({});
  const [running, setRunning] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [result, setResult] = createSignal<Record<string, unknown> | null>(null);

  function loadPreset(p: Preset) {
    setStateText(JSON.stringify(p.state, null, 2));
    setPreset(p.questions);
    setResult(null);
    setError(null);
  }

  function clearForm() {
    setStateText("");
    setPreset({});
    setResult(null);
    setError(null);
  }

  async function run() {
    const m = model();
    if (!m) return;
    setRunning(true);
    setError(null);
    try {
      const state = JSON.parse(stateText() || "null");
      const qs = questions();
      if (Object.keys(qs).length === 0) throw new Error("add at least one question");
      const raw = m.model.ask(JSON.stringify(state), JSON.stringify(qs));
      setResult(JSON.parse(raw));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div class="mx-auto w-full max-w-5xl px-5 py-10">
      <p class="font-mono text-xs text-accent">100% client-side · runs in your browser via WebAssembly</p>
      <h1 class="mt-2 text-3xl font-semibold tracking-tight text-ink">Playground</h1>
      <p class="mt-2 max-w-2xl leading-relaxed text-muted">
        Ask <Mono>choice</Mono>/<Mono>score</Mono>/<Mono>noul</Mono> questions over a state object —
        build them with the form below, or paste JSON directly. Every question is answered
        against the whole state in a single batched forward pass.
      </p>

      <div class="mt-6">
        <ModelPicker models={CHECKPOINT_MODELS} onLoaded={setModel} />
      </div>

      <div class="mt-6 flex flex-wrap items-center gap-2">
        <span class="text-xs text-faint">Presets:</span>
        <For each={PRESETS}>
          {(p) => (
            <Button variant="secondary" size="sm" onClick={() => loadPreset(p)}>
              {p.label}
            </Button>
          )}
        </For>
        <Button variant="secondary" size="sm" onClick={clearForm}>
          Clear
        </Button>
      </div>

      <div class="mt-6">
        <label class="block text-sm text-muted">
          State <span class="text-faint">(JSON, or plain text — fed to the model as context)</span>
          <JsonEditor value={stateText()} onInput={setStateText} rows={10} />
        </label>
      </div>

      <div class="mt-6">
        <p class="text-sm text-muted">Questions</p>
        <div class="mt-1">
          <QuestionBuilder preset={preset} onChange={setQuestions} />
        </div>
      </div>

      <div class="mt-4 flex items-center gap-3">
        <Button variant="primary" disabled={!model() || running()} onClick={run}>
          {running() ? "Asking…" : "Ask"}
        </Button>
        <Show when={!model()}>
          <span class="text-xs text-faint">Load a checkpoint above first.</span>
        </Show>
      </div>

      <Show when={error()}>
        <p class="mt-4 text-sm text-danger">{error()}</p>
      </Show>

      <Show when={result()}>
        {(r) => (
          <div class="mt-6">
            <div class="mb-3 flex items-center gap-2">
              <Badge tone="accent">Answers</Badge>
            </div>
            <AnswerView answers={r() as any} />
          </div>
        )}
      </Show>
    </div>
  );
}
