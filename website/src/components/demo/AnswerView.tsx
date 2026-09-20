import { For, Show } from "solid-js";
import { Badge } from "../ui";

/** The JSON shape `agent::answer_to_json` produces — one per question,
 * exactly what `laya jev` writes and `WasmAgent::ask` returns. */
interface ChoiceAnswer {
  type: "choice";
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
  act_probability: number;
}
interface ScoreAnswer {
  type: "score";
  score: number;
  legend: Record<string, string>;
  probabilities: Record<string, number>;
  confidence: number;
  act_probability: number;
}
interface NoulAnswer {
  type: "noul";
  noul: number;
  act_probability: number;
}
type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

function pct(x: number): string {
  return `${Math.round(x * 100)}%`;
}

function Bar(props: { label: string; value: number; highlight?: boolean }) {
  return (
    <div class="flex items-center gap-2 text-xs">
      <span class={`w-28 shrink-0 truncate font-mono ${props.highlight ? "text-ink font-medium" : "text-muted"}`}>
        {props.label}
      </span>
      <div class="h-3 flex-1 overflow-hidden rounded-full bg-surface-3">
        <div
          class={`h-full rounded-full ${props.highlight ? "bg-accent" : "bg-line-strong"}`}
          style={{ width: pct(props.value) }}
        />
      </div>
      <span class="w-10 shrink-0 text-right font-mono text-faint">{pct(props.value)}</span>
    </div>
  );
}

function Meta(props: { confidence?: number; actProbability: number }) {
  return (
    <div class="mt-3 flex flex-wrap gap-2">
      <Show when={props.confidence !== undefined}>
        <Badge tone="neutral">confidence {pct(props.confidence!)}</Badge>
      </Show>
      <Badge tone={props.actProbability > 0.5 ? "warn" : "neutral"}>abstain {pct(props.actProbability)}</Badge>
    </div>
  );
}

function ChoiceView(props: { qid: string; answer: ChoiceAnswer }) {
  const entries = () => Object.entries(props.answer.probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <div class="rounded-xl border border-line bg-surface p-4">
      <div class="flex items-center justify-between gap-2">
        <p class="font-mono text-sm text-ink">{props.qid}</p>
        <Badge tone="accent">{props.answer.choice}</Badge>
      </div>
      <div class="mt-3 space-y-1.5">
        <For each={entries()}>{([key, p]) => <Bar label={key} value={p} highlight={key === props.answer.choice} />}</For>
      </div>
      <Meta confidence={props.answer.confidence} actProbability={props.answer.act_probability} />
    </div>
  );
}

function ScoreView(props: { qid: string; answer: ScoreAnswer }) {
  const entries = () =>
    Object.entries(props.answer.probabilities)
      .sort((a, b) => Number(a[0]) - Number(b[0]))
      .map(([i, p]) => ({ i, p, label: props.answer.legend[i] ?? i }));
  const top = () => entries().reduce((a, b) => (b.p > a.p ? b : a), entries()[0]);
  return (
    <div class="rounded-xl border border-line bg-surface p-4">
      <div class="flex items-center justify-between gap-2">
        <p class="font-mono text-sm text-ink">{props.qid}</p>
        <Badge tone="accent">score {props.answer.score.toFixed(2)}</Badge>
      </div>
      <div class="mt-3 space-y-1.5">
        <For each={entries()}>{(e) => <Bar label={e.label} value={e.p} highlight={e.i === top()?.i} />}</For>
      </div>
      <Meta confidence={props.answer.confidence} actProbability={props.answer.act_probability} />
    </div>
  );
}

function NoulView(props: { qid: string; answer: NoulAnswer }) {
  return (
    <div class="rounded-xl border border-line bg-surface p-4">
      <div class="flex items-center justify-between gap-2">
        <p class="font-mono text-sm text-ink">{props.qid}</p>
        <Badge tone={props.answer.noul >= 0.5 ? "accent" : "neutral"}>{props.answer.noul >= 0.5 ? "true" : "false"}</Badge>
      </div>
      <div class="mt-3">
        <Bar label="P(true)" value={props.answer.noul} highlight />
      </div>
      <Meta actProbability={props.answer.act_probability} />
    </div>
  );
}

/** Renders one card per question, laid out for its answer type — a bar
 * chart of option probabilities for `choice`, a weighted score plus its
 * distribution over levels for `score`, a single P(true) bar for `noul`. */
export function AnswerView(props: { answers: Record<string, Answer> }) {
  return (
    <div class="grid gap-3 sm:grid-cols-2">
      <For each={Object.entries(props.answers)}>
        {([qid, answer]) => (
          <Show when={answer.type === "choice"} fallback={
            <Show when={answer.type === "score"} fallback={<NoulView qid={qid} answer={answer as NoulAnswer} />}>
              <ScoreView qid={qid} answer={answer as ScoreAnswer} />
            </Show>
          }>
            <ChoiceView qid={qid} answer={answer as ChoiceAnswer} />
          </Show>
        )}
      </For>
    </div>
  );
}
