import { highlight } from "../../lib/highlight";

/** An editable textarea with live JSON syntax highlighting: a transparent
 * textarea sits on top of a highlighted <pre>, sharing the same font metrics
 * and scroll offset so the caret lines up with the colored text underneath. */
export function JsonEditor(props: { value: string; onInput: (v: string) => void; rows?: number }) {
  let pre: HTMLPreElement | undefined;

  function syncScroll(e: Event) {
    const ta = e.currentTarget as HTMLTextAreaElement;
    if (pre) {
      pre.scrollTop = ta.scrollTop;
      pre.scrollLeft = ta.scrollLeft;
    }
  }

  return (
    <div class="relative mt-1 w-full rounded-md border border-line bg-surface-2 focus-within:border-accent">
      <pre
        ref={pre}
        aria-hidden="true"
        class="pointer-events-none m-0 overflow-auto whitespace-pre-wrap break-words px-2 py-1.5 font-mono text-xs leading-normal"
        style={{ height: `${(props.rows ?? 10) * 1.25 + 0.75}rem` }}
      >
        <code innerHTML={highlight(props.value, "json") + "\n"} />
      </pre>
      <textarea
        value={props.value}
        onInput={(e) => props.onInput(e.currentTarget.value)}
        onScroll={syncScroll}
        spellcheck={false}
        rows={props.rows ?? 10}
        class="absolute inset-0 h-full w-full resize-none whitespace-pre-wrap break-words bg-transparent px-2 py-1.5 font-mono text-xs leading-normal text-transparent caret-ink outline-none"
      />
    </div>
  );
}
