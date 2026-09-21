import { createSignal, createStore, For, Show, createEffect } from "solid-js";
import { Button } from "../ui";
import { JsonView } from "./JsonView";

/** The native laya question schema (`batching::RawQuestion`) — the same
 * shape `laya answer`'s input files and `WasmAgent::ask`'s `questions_json`
 * both use, so this builder needs no intermediate conversion. */
export type QuestionType = "choice" | "score" | "noul";

export interface ChoiceQuestion {
  type: "choice";
  instructions: string;
  criteria: Record<string, string | null>;
}
export interface ScoreQuestion {
  type: "score";
  instructions: string;
  criteria: string[];
}
export interface NoulQuestion {
  type: "noul";
  instructions: string;
  criteria?: { true?: string; false?: string };
}
export type RawQuestion = ChoiceQuestion | ScoreQuestion | NoulQuestion;
export type QuestionSet = Record<string, RawQuestion>;

/* ------------------------------------------------------------------ */
/* Internal editable card state — richer than RawQuestion so half-typed  */
/* rows (an option with no description yet, an empty new level) don't    */
/* have to round-trip through the serialized shape on every keystroke.   */
/* ------------------------------------------------------------------ */

let nextId = 0;
const uid = () => `r${nextId++}`;

interface ChoiceRow {
  id: string;
  key: string;
  desc: string;
}
interface ScoreRow {
  id: string;
  text: string;
}
interface Card {
  id: string;
  qid: string;
  type: QuestionType;
  instructions: string;
  choiceRows: ChoiceRow[];
  scoreRows: ScoreRow[];
  noulTrue: string;
  noulFalse: string;
}

function emptyCard(qid: string, type: QuestionType = "choice"): Card {
  return {
    id: uid(),
    qid,
    type,
    instructions: "",
    choiceRows:
      type === "choice"
        ? [
            { id: uid(), key: "yes", desc: "" },
            { id: uid(), key: "no", desc: "" },
          ]
        : [],
    scoreRows: type === "score" ? [{ id: uid(), text: "" }, { id: uid(), text: "" }] : [],
    noulTrue: "",
    noulFalse: "",
  };
}

function cardToRaw(c: Card): RawQuestion {
  if (c.type === "choice") {
    const criteria: Record<string, string | null> = {};
    for (const row of c.choiceRows) {
      if (row.key.trim()) criteria[row.key.trim()] = row.desc.trim() || null;
    }
    return { type: "choice", instructions: c.instructions, criteria };
  }
  if (c.type === "score") {
    return { type: "score", instructions: c.instructions, criteria: c.scoreRows.map((r) => r.text) };
  }
  const criteria: { true?: string; false?: string } = {};
  if (c.noulTrue.trim()) criteria.true = c.noulTrue.trim();
  if (c.noulFalse.trim()) criteria.false = c.noulFalse.trim();
  return { type: "noul", instructions: c.instructions, criteria: Object.keys(criteria).length ? criteria : undefined };
}

function rawToCards(qs: QuestionSet): Card[] {
  return Object.entries(qs).map(([qid, q]) => {
    const card = emptyCard(qid, q.type);
    card.instructions = typeof q.instructions === "string" ? q.instructions : JSON.stringify(q.instructions);
    if (q.type === "choice") {
      card.choiceRows = Object.entries(q.criteria ?? {}).map(([key, desc]) => ({ id: uid(), key, desc: desc ?? "" }));
      if (card.choiceRows.length === 0) card.choiceRows = [{ id: uid(), key: "", desc: "" }];
    } else if (q.type === "score") {
      card.scoreRows = (q.criteria ?? []).map((text) => ({ id: uid(), text }));
      if (card.scoreRows.length === 0) card.scoreRows = [{ id: uid(), text: "" }];
    } else {
      card.noulTrue = q.criteria?.true ?? "";
      card.noulFalse = q.criteria?.false ?? "";
    }
    return card;
  });
}

function serialize(cards: readonly Card[]): QuestionSet {
  const out: QuestionSet = {};
  for (const c of cards) {
    if (c.qid.trim()) out[c.qid.trim()] = cardToRaw(c);
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Field primitives.                                                   */
/* ------------------------------------------------------------------ */

function fieldLabel(text: string) {
  return <span class="text-xs text-faint">{text}</span>;
}

const inputClass =
  "mt-1 w-full rounded-md border border-line bg-surface-2 px-2 py-1.5 text-sm text-ink outline-none focus:border-accent";

function IconButton(props: { onClick: () => void; title: string; danger?: boolean; children: any }) {
  return (
    <button
      type="button"
      title={props.title}
      onClick={props.onClick}
      class={`inline-flex h-6 w-6 items-center justify-center rounded-md transition-colors ${
        props.danger ? "text-faint hover:bg-danger/10 hover:text-danger" : "text-faint hover:bg-surface-3 hover:text-ink"
      }`}
    >
      {props.children}
    </button>
  );
}

function TrashIcon() {
  return (
    <svg viewBox="0 0 24 24" class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="1.8">
      <path stroke-linecap="round" stroke-linejoin="round" d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2m2 0-1 13a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1L6 7h12Z" />
    </svg>
  );
}
function UpIcon() {
  return (
    <svg viewBox="0 0 24 24" class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="1.8">
      <path stroke-linecap="round" stroke-linejoin="round" d="m6 15 6-6 6 6" />
    </svg>
  );
}
function DownIcon() {
  return (
    <svg viewBox="0 0 24 24" class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="1.8">
      <path stroke-linecap="round" stroke-linejoin="round" d="m6 9 6 6 6-6" />
    </svg>
  );
}

const TYPE_TABS: { value: QuestionType; label: string; hint: string }[] = [
  { value: "choice", label: "choice", hint: "pick one of several named options" },
  { value: "score", label: "score", hint: "an ordered scale of levels" },
  { value: "noul", label: "noul", hint: "yes/no, as a probability" },
];

/* ------------------------------------------------------------------ */
/* One question card.                                                  */
/* ------------------------------------------------------------------ */

function QuestionCard(props: { card: Card; onMutate: (fn: (c: Card) => void) => void; onRemove: () => void }) {
  // `onMutate` reaches into this card's slot in the parent store and hands
  // the callback a mutable draft: `c.qid = v`, `c.choiceRows.push(...)`.
  // That's an in-place write, not a new object, so <For> never sees a
  // changed reference and never remounts the row — which is what keeps
  // focus in the input across keystrokes. Rebuilding cards/rows via spread
  // (the old approach) handed <For> a fresh object per keystroke instead.

  function setType(type: QuestionType) {
    if (type === props.card.type) return;
    const fresh = emptyCard(props.card.qid, type);
    props.onMutate((c) => {
      c.type = type;
      c.choiceRows = fresh.choiceRows;
      c.scoreRows = fresh.scoreRows;
      c.noulTrue = "";
      c.noulFalse = "";
    });
  }

  function addChoiceRow() {
    props.onMutate((c) => {
      c.choiceRows.push({ id: uid(), key: "", desc: "" });
    });
  }
  function updateChoiceRow(id: string, key: "key" | "desc", value: string) {
    props.onMutate((c) => {
      const row = c.choiceRows.find((r) => r.id === id);
      if (row) row[key] = value;
    });
  }
  function removeChoiceRow(id: string) {
    props.onMutate((c) => {
      const i = c.choiceRows.findIndex((r) => r.id === id);
      if (i >= 0) c.choiceRows.splice(i, 1);
    });
  }

  function addScoreRow() {
    props.onMutate((c) => {
      c.scoreRows.push({ id: uid(), text: "" });
    });
  }
  function updateScoreRow(id: string, text: string) {
    props.onMutate((c) => {
      const row = c.scoreRows.find((r) => r.id === id);
      if (row) row.text = text;
    });
  }
  function removeScoreRow(id: string) {
    props.onMutate((c) => {
      const i = c.scoreRows.findIndex((r) => r.id === id);
      if (i >= 0) c.scoreRows.splice(i, 1);
    });
  }
  function moveScoreRow(id: string, dir: -1 | 1) {
    props.onMutate((c) => {
      const i = c.scoreRows.findIndex((r) => r.id === id);
      const j = i + dir;
      if (i < 0 || j < 0 || j >= c.scoreRows.length) return;
      const tmp = c.scoreRows[i];
      c.scoreRows[i] = c.scoreRows[j];
      c.scoreRows[j] = tmp;
    });
  }

  return (
    <div class="rounded-xl border border-line bg-surface p-4">
      <div class="flex flex-wrap items-start justify-between gap-3">
        <label class="min-w-0 flex-1">
          {fieldLabel("question id")}
          <input
            class={`${inputClass} font-mono`}
            value={props.card.qid}
            onInput={(e) => {
              const v = e.currentTarget.value;
              props.onMutate((c) => {
                c.qid = v;
              });
            }}
            placeholder="e.g. department"
          />
        </label>
        <div class="flex items-center gap-1 rounded-lg border border-line bg-surface-2 p-0.5">
          <For each={TYPE_TABS}>
            {(t) => (
              <button
                type="button"
                title={t.hint}
                onClick={() => setType(t.value)}
                class={`rounded-md px-2.5 py-1 font-mono text-xs transition-colors ${
                  props.card.type === t.value ? "bg-accent text-on-accent" : "text-muted hover:text-ink"
                }`}
              >
                {t.label}
              </button>
            )}
          </For>
        </div>
        <IconButton title="Remove question" danger onClick={props.onRemove}>
          <TrashIcon />
        </IconButton>
      </div>

      <label class="mt-3 block">
        {fieldLabel("instructions")}
        <textarea
          class={`${inputClass} resize-y`}
          rows={2}
          value={props.card.instructions}
          onInput={(e) => {
            const v = e.currentTarget.value;
            props.onMutate((c) => {
              c.instructions = v;
            });
          }}
          placeholder="What should the model decide?"
        />
      </label>

      <Show when={props.card.type === "choice"}>
        <div class="mt-3">
          {fieldLabel("options")}
          <div class="mt-1 space-y-1.5">
            <For each={props.card.choiceRows}>
              {(row) => (
                <div class="flex items-center gap-1.5">
                  <input
                    class={`${inputClass} mt-0 w-32 shrink-0 font-mono`}
                    value={row.key}
                    onInput={(e) => updateChoiceRow(row.id, "key", e.currentTarget.value)}
                    placeholder="option key"
                  />
                  <input
                    class={`${inputClass} mt-0 flex-1`}
                    value={row.desc}
                    onInput={(e) => updateChoiceRow(row.id, "desc", e.currentTarget.value)}
                    placeholder="description (optional)"
                  />
                  <IconButton title="Remove option" danger onClick={() => removeChoiceRow(row.id)}>
                    <TrashIcon />
                  </IconButton>
                </div>
              )}
            </For>
          </div>
          <Button size="sm" variant="secondary" class="mt-2" onClick={addChoiceRow}>
            + Add option
          </Button>
        </div>
      </Show>

      <Show when={props.card.type === "score"}>
        <div class="mt-3">
          {fieldLabel("levels, low to high")}
          <div class="mt-1 space-y-1.5">
            <For each={props.card.scoreRows}>
              {(row, i) => (
                <div class="flex items-center gap-1.5">
                  <span class="w-5 shrink-0 text-right font-mono text-xs text-faint">{i()}</span>
                  <input
                    class={`${inputClass} mt-0 flex-1`}
                    value={row.text}
                    onInput={(e) => updateScoreRow(row.id, e.currentTarget.value)}
                    placeholder="level description"
                  />
                  <IconButton title="Move up" onClick={() => moveScoreRow(row.id, -1)}>
                    <UpIcon />
                  </IconButton>
                  <IconButton title="Move down" onClick={() => moveScoreRow(row.id, 1)}>
                    <DownIcon />
                  </IconButton>
                  <IconButton title="Remove level" danger onClick={() => removeScoreRow(row.id)}>
                    <TrashIcon />
                  </IconButton>
                </div>
              )}
            </For>
          </div>
          <Button size="sm" variant="secondary" class="mt-2" onClick={addScoreRow}>
            + Add level
          </Button>
        </div>
      </Show>

      <Show when={props.card.type === "noul"}>
        <div class="mt-3 grid gap-2 sm:grid-cols-2">
          <label>
            {fieldLabel("true means (optional)")}
            <input
              class={inputClass}
              value={props.card.noulTrue}
              onInput={(e) => {
                const v = e.currentTarget.value;
                props.onMutate((c) => {
                  c.noulTrue = v;
                });
              }}
              placeholder="yes, the statement holds"
            />
          </label>
          <label>
            {fieldLabel("false means (optional)")}
            <input
              class={inputClass}
              value={props.card.noulFalse}
              onInput={(e) => {
                const v = e.currentTarget.value;
                props.onMutate((c) => {
                  c.noulFalse = v;
                });
              }}
              placeholder="no, the statement does not hold"
            />
          </label>
        </div>
      </Show>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The builder.                                                        */
/* ------------------------------------------------------------------ */

export function QuestionBuilder(props: {
  preset: () => QuestionSet | null;
  onChange: (qs: QuestionSet) => void;
}) {
  // A store, not a signal: edits mutate a card/row in place (see
  // QuestionCard's onMutate) instead of rebuilding the array with fresh
  // object references on every keystroke, which is what was remounting the
  // edited row (and stealing focus) on each typed letter.
  const [cards, setCards] = createStore<Card[]>([emptyCard("question_1")]);
  const [importText, setImportText] = createSignal("");
  const [importError, setImportError] = createSignal<string | null>(null);

  // This Solid version requires createEffect(computeFn, effectFn): the
  // compute function is what's tracked, and the effect function only runs
  // when its return value actually changes — exactly the "new preset
  // object, load it once" behavior wanted here, with no manual bookkeeping.
  createEffect(
    () => props.preset(),
    (preset) => {
      if (preset) setCards(() => rawToCards(preset));
    },
  );

  createEffect(
    () => serialize(cards),
    (qs) => {
      // Braced body: an effect function may only return undefined or an
      // actual cleanup function, and onChange's return value (whatever the
      // caller's setter happens to hand back) isn't either.
      props.onChange(qs);
    },
  );

  function addCard() {
    setCards((draft) => {
      draft.push(emptyCard(`question_${draft.length + 1}`));
    });
  }
  function mutateCard(id: string, fn: (c: Card) => void) {
    setCards((draft) => {
      const card = draft.find((c) => c.id === id);
      if (card) fn(card);
    });
  }
  function removeCard(id: string) {
    setCards((draft) => {
      const i = draft.findIndex((c) => c.id === id);
      if (i >= 0) draft.splice(i, 1);
    });
  }

  function tryImport() {
    setImportError(null);
    try {
      const parsed = JSON.parse(importText()) as QuestionSet;
      setCards(() => rawToCards(parsed));
      setImportText("");
    } catch (e) {
      setImportError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div>
      <div class="space-y-3">
        <For each={cards}>
          {(card) => (
            <QuestionCard card={card} onMutate={(fn) => mutateCard(card.id, fn)} onRemove={() => removeCard(card.id)} />
          )}
        </For>
      </div>

      <Button variant="secondary" class="mt-3" onClick={addCard}>
        + Add question
      </Button>

      <details class="mt-4 rounded-xl border border-line bg-surface p-4">
        <summary class="cursor-pointer text-sm font-medium text-ink">Raw JSON (view / import)</summary>
        <div class="mt-3">
          <JsonView value={serialize(cards)} />
        </div>
        <label class="mt-3 block">
          {fieldLabel("paste questions JSON to replace everything above")}
          <textarea
            class={`${inputClass} font-mono`}
            rows={4}
            value={importText()}
            onInput={(e) => setImportText(e.currentTarget.value)}
            placeholder={'{ "department": { "type": "choice", "instructions": "...", "criteria": { "billing": null } } }'}
          />
        </label>
        <div class="mt-2 flex items-center gap-3">
          <Button size="sm" variant="secondary" disabled={!importText().trim()} onClick={tryImport}>
            Import
          </Button>
          <Show when={importError()}>
            <span class="text-xs text-danger">{importError()}</span>
          </Show>
        </div>
      </details>
    </div>
  );
}
