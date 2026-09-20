import { For, Show, createSignal } from "solid-js";
import { highlight } from "../lib/highlight";
import { CODE_WIDTH } from "./docs/Prose";

/**
 * A multi-line command block with a copy button. The whole block copies as
 * one command; the button shows a check for a moment on success. Each line
 * is bash-highlighted, the leading prompt excluded so it stays plain.
 */
export function CopyBlock(props: { command: string; prompt?: string }) {
  const [copied, setCopied] = createSignal(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(props.command);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard denied: the text remains selectable */
    }
  };

  const lines = () => props.command.split("\n");

  return (
    <div class={`relative min-w-0 ${CODE_WIDTH} overflow-hidden rounded-xl border border-line bg-surface`}>
      <pre class="overflow-x-auto px-4 py-3.5 pr-12 font-mono text-[0.8125rem] leading-relaxed text-muted">
        <code class="language-bash">
          <For each={lines()}>
            {(line, index) => (
              <>
                <span class="select-none text-faint">{props.prompt ?? "$ "}</span>
                <span innerHTML={highlight(line, "bash")} />
                {index() < lines().length - 1 ? "\n" : ""}
              </>
            )}
          </For>
        </code>
      </pre>
      <button
        type="button"
        onClick={copy}
        aria-label="Copy commands"
        class="absolute right-3 top-3 inline-flex h-6 w-6 items-center justify-center rounded-md text-faint transition-colors hover:bg-surface-2 hover:text-ink"
      >
        <Show
          when={copied()}
          fallback={
            <svg viewBox="0 0 24 24" class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="1.8">
              <rect x="9" y="9" width="11" height="11" rx="2" />
              <path d="M5 15V5a2 2 0 0 1 2-2h8" stroke-linecap="round" />
            </svg>
          }
        >
          <svg viewBox="0 0 24 24" class="h-3.5 w-3.5 text-success" fill="none" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="m5 13 4 4L19 7" />
          </svg>
        </Show>
      </button>
    </div>
  );
}
