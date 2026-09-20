import { createEffect, createMemo, createSignal, For, Show } from "solid-js";
import { Badge, Button } from "../ui";
import type { ModelDef } from "../../lib/models";
import {
  type DownloadProgress,
  availableLocally,
  clearModelCache,
  currentModel,
  ensureLocalLibraryResumeAttempted,
  fsAccessSupported,
  grantPendingLibraryAccess,
  localLibrary,
  pendingLibrary,
  persistedState,
  preferredSource,
  selectLocalLibrary,
  selectLocalModel,
  selectModel,
  unloadModel,
  type LoadedModel,
} from "../../lib/laya";

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function ProgressBar(props: { progress: DownloadProgress | null }) {
  const pct = () => {
    const p = props.progress;
    if (!p || !p.total) return null;
    return Math.round((p.loaded / p.total) * 100);
  };
  return (
    <Show when={props.progress}>
      {(p) => (
        <div class="mt-3">
          <div class="flex justify-between text-xs text-faint">
            <span>
              {p().file} ({p().fileIndex + 1}/{p().fileCount})
            </span>
            <span>
              {fmtBytes(p().loaded)}
              {p().total ? ` / ${fmtBytes(p().total!)}` : ""}
            </span>
          </div>
          <div class="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
            <div class="h-full rounded-full bg-accent transition-all duration-150" style={{ width: `${pct() ?? 15}%` }} />
          </div>
        </div>
      )}
    </Show>
  );
}

/** Picks a checkpoint — either downloaded from Hugging Face (cached in the
 * browser) or read from a local library directory — and hands the loaded
 * model up via `onLoaded`. Shown as two columns, one source each, so both
 * are visible without a toggle.
 *
 * The local library directory is picked once, holding one subdirectory per
 * checkpoint named after its Hugging Face repo (e.g. `laya-typed-decisions`);
 * every demo tab then lists whichever of its models are found inside it,
 * without re-picking a folder per tab or per model. */
export function ModelPicker(props: { models: ModelDef[]; onLoaded: (m: LoadedModel) => void }) {
  const [selected, setSelected] = createSignal(props.models[0]);
  const [busy, setBusy] = createSignal(false);
  const [progress, setProgress] = createSignal<DownloadProgress | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const [loadedKey, setLoadedKey] = createSignal<string | null>(null);

  const localMatches = createMemo(() => {
    const lib = localLibrary();
    return lib ? availableLocally(lib, props.models) : [];
  });

  async function load() {
    setBusy(true);
    setError(null);
    setProgress(null);
    try {
      const def = selected();
      const result = await selectModel(def, setProgress);
      setLoadedKey(def.key);
      props.onLoaded(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function loadLocal(def: ModelDef) {
    setBusy(true);
    setError(null);
    setProgress(null);
    try {
      const result = await selectLocalModel(def, setProgress);
      setLoadedKey(def.key);
      props.onLoaded(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function pickLibraryFolder() {
    setError(null);
    try {
      const library = await selectLocalLibrary();
      if (!library) return; // user dismissed the picker
      if (availableLocally(library, props.models).length === 0) {
        const names = props.models.flatMap((m) => [m.hfRepo.split("/").pop()!, m.subfolder]);
        setError(
          `No matching checkpoints found in ${library.rootName}. Expected a subdirectory named after one of: ${Array.from(
            new Set(names),
          ).join(", ")}.`,
        );
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function grantAccess() {
    setError(null);
    try {
      if (!(await grantPendingLibraryAccess())) setError("Access wasn't granted.");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function clearCache() {
    setError(null);
    try {
      await clearModelCache();
      if (preferredSource() === "hf") {
        unloadModel();
        setLoadedKey(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  // Resume/auto-select, in priority order: a local library match (even one
  // that only just appeared — picked in another tab, or found by the silent
  // resume check kicked off below — wins over an already-loaded Hugging
  // Face model, since it's already on disk and costs nothing to use) beats
  // the Hugging Face model remembered from last session, which beats
  // leaving the picker empty. Runs whenever the local library or the shared
  // current model changes; each branch is a no-op once it's already
  // satisfied.
  createEffect(
    () => [localLibrary(), currentModel()] as const,
    ([lib, existing]) => {
      // Not gated on last session's preferred source — a local directory
      // always wins when both have the checkpoint, so it's worth checking
      // even if Hugging Face was picked last time. Idempotent; only the
      // first call in a session does anything.
      ensureLocalLibraryResumeAttempted();

      if (busy()) return;

      const matches = lib ? availableLocally(lib, props.models) : [];
      if (matches.length > 0 && !(loadedKey() && preferredSource() === "local")) {
        // Always the largest one found — the best-quality checkpoint on
        // hand, since it's already local and costs nothing to use.
        void loadLocal(matches.reduce((a, b) => (b.approxSizeMb > a.approxSizeMb ? b : a)));
        return;
      }
      if (matches.length > 0) return; // already showing the right local model

      if (loadedKey()) return; // already resolved for this picker otherwise
      if (existing) {
        setLoadedKey(existing.def.key);
        return;
      }
      const persisted = persistedState();
      if (persisted.source === "hf" && persisted.hfModelKey) {
        const def = props.models.find((m) => m.key === persisted.hfModelKey);
        if (def) {
          setSelected(def);
          void load();
        }
      }
    },
  );

  return (
    <div class="rounded-xl border border-line bg-surface p-4">
      <div class="grid gap-4 md:grid-cols-2">
        {/* Hugging Face */}
        <div class="md:border-r md:border-line md:pr-4">
          <div class="flex items-center gap-2">
            <p class="text-sm font-medium text-ink">Hugging Face</p>
            <Show when={loadedKey() === selected().key && preferredSource() === "hf" && !busy()}>
              <Badge tone="accent">Loaded</Badge>
            </Show>
          </div>
          <div class="mt-3 flex flex-wrap items-center gap-2">
            <For each={props.models}>
              {(m) => (
                <button
                  type="button"
                  disabled={busy()}
                  onClick={() => setSelected(m)}
                  class={`rounded-lg border px-3 py-2 text-left text-sm transition-colors ${
                    selected().key === m.key
                      ? "border-accent bg-accent-soft text-ink"
                      : "border-line bg-surface-2 text-muted hover:border-line-strong hover:text-ink"
                  }`}
                >
                  <div class="font-medium">{m.label}</div>
                  <div class="mt-0.5 text-xs text-faint">~{m.approxSizeMb} MB</div>
                </button>
              )}
            </For>
          </div>
          <p class="mt-3 text-sm text-muted">{selected().description}</p>
          <p class="mt-1 font-mono text-xs text-faint">{selected().hfRepo}</p>
          <div class="mt-4 flex items-center gap-3">
            <Button variant="primary" disabled={busy()} onClick={load}>
              {busy() ? "Loading…" : loadedKey() === selected().key && preferredSource() === "hf" ? "Reload" : "Download & load"}
            </Button>
            <button
              type="button"
              disabled={busy()}
              onClick={clearCache}
              class="text-xs text-muted underline decoration-dotted hover:text-ink disabled:opacity-40"
            >
              clear downloaded cache
            </button>
          </div>
        </div>

        {/* Local directory */}
        <div>
          <div class="flex items-center gap-2">
            <p class="text-sm font-medium text-ink">Local directory</p>
            <Show when={loadedKey() && preferredSource() === "local" && !busy()}>
              <Badge tone="accent">Loaded</Badge>
            </Show>
          </div>

          <Show
            when={fsAccessSupported()}
            fallback={
              <p class="mt-3 text-sm text-danger">
                This browser doesn't support the File System Access API needed to load a local directory.
                Try Chrome or Edge.
              </p>
            }
          >
          <Show
            when={localLibrary()}
            fallback={
              <div class="mt-3">
                <Show
                  when={pendingLibrary()}
                  fallback={
                    <p class="text-sm text-muted">
                      Pick a directory with one subdirectory per checkpoint — either a folder of
                      individually-cloned repos (e.g.{" "}
                      <code class="font-mono text-xs">laya-typed-decisions/</code>) or a single
                      clone of the <code class="font-mono text-xs">convaiinnovations/laya</code>{" "}
                      family repo itself (same layout the CLI's <code class="font-mono text-xs">models_root</code>{" "}
                      expects, e.g. <code class="font-mono text-xs">~/laya</code> with its own{" "}
                      <code class="font-mono text-xs">typed-decisions/</code>/
                      <code class="font-mono text-xs">multilingual/</code> subfolders). Each
                      checkpoint directory should hold{" "}
                      <code class="font-mono text-xs">rl_agent_config.json</code>,{" "}
                      <code class="font-mono text-xs">encoder/config.json</code>,{" "}
                      <code class="font-mono text-xs">tokenizer/tokenizer.json</code>,{" "}
                      <code class="font-mono text-xs">tokenizer/tokenizer_config.json</code> and{" "}
                      <code class="font-mono text-xs">model.safetensors</code>.
                    </p>
                  }
                >
                  {(pending) => (
                    <>
                      <p class="mb-3 rounded-lg border border-accent-line bg-accent-soft px-3 py-2 text-sm text-ink">
                        Resuming <span class="font-mono text-xs">{pending().rootName}/</span> from last time — grant
                        access to reload it, no need to pick it again.
                      </p>
                      <Button variant="primary" disabled={busy()} onClick={grantAccess}>
                        Grant access
                      </Button>
                    </>
                  )}
                </Show>
                <Button class="mt-3" variant={pendingLibrary() ? "secondary" : "primary"} disabled={busy()} onClick={pickLibraryFolder}>
                  {pendingLibrary() ? "Choose a different folder…" : "Choose folder…"}
                </Button>
              </div>
            }
          >
            {(lib) => (
              <div class="mt-3">
                <div class="flex items-center justify-between">
                  <p class="font-mono text-xs text-faint">{lib().rootName}/</p>
                  <button
                    type="button"
                    disabled={busy()}
                    onClick={pickLibraryFolder}
                    class="text-xs text-muted underline decoration-dotted hover:text-ink"
                  >
                    change folder
                  </button>
                </div>
                <div class="mt-3 flex flex-wrap items-center gap-2">
                  <For each={props.models}>
                    {(m) => {
                      const foundHere = () => localMatches().some((x) => x.key === m.key);
                      return (
                        <button
                          type="button"
                          disabled={busy() || !foundHere()}
                          onClick={() => loadLocal(m)}
                          title={foundHere() ? undefined : `not found in ${lib().rootName}/`}
                          class={`rounded-lg border px-3 py-2 text-left text-sm transition-colors disabled:opacity-40 ${
                            loadedKey() === m.key && preferredSource() === "local"
                              ? "border-accent bg-accent-soft text-ink"
                              : "border-line bg-surface-2 text-muted hover:border-line-strong hover:text-ink"
                          }`}
                        >
                          <div class="font-medium">{m.label}</div>
                          <div class="mt-0.5 text-xs text-faint">{foundHere() ? "found" : "not found"}</div>
                        </button>
                      );
                    }}
                  </For>
                </div>
              </div>
            )}
          </Show>
          </Show>
        </div>
      </div>

      <ProgressBar progress={busy() ? progress() : null} />

      <Show when={error()}>
        <p class="mt-3 text-sm text-danger">{error()}</p>
      </Show>
    </div>
  );
}
