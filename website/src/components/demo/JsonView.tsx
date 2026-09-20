import { highlight } from "../../lib/highlight";

/** Pretty-printed, syntax-highlighted JSON output panel. */
export function JsonView(props: { value: unknown }) {
  const text = () => JSON.stringify(props.value, null, 2);
  return (
    <pre class="max-h-[32rem] overflow-auto rounded-xl border border-line bg-code-bg px-4 py-3.5 font-mono text-[0.8125rem] leading-relaxed">
      <code innerHTML={highlight(text(), "json")} />
    </pre>
  );
}
