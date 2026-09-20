/** Loads laya-rs's wasm module and checkpoints fetched straight from the
 * Hugging Face Hub, entirely client-side — no backend involved.
 *
 * Checkpoint files are cached with the browser Cache API (origin-scoped, so
 * they survive reloads and switching between demo pages) keyed by their HF
 * URL, since HF's `resolve/main/...` content is immutable per commit.
 */

import { createSignal } from "solid-js";
import { CHECKPOINT_FILES, hfFileUrl, type ModelDef } from "./models";
import {
  fsAccessSupported,
  getSubdirectory,
  listSubdirectories,
  loadRememberedDirectory,
  pickDirectory,
  queryReadPermission,
  readRelativeFile,
  rememberDirectory,
  requestReadPermission,
} from "./localFs";

export { fsAccessSupported };

export interface WasmAgent {
  ask(stateJson: string, questionsJson: string): string;
  free(): void;
}

type WasmExports = {
  default: (module_or_path?: unknown) => Promise<unknown>;
  WasmAgent: {
    load(
      weights: Uint8Array,
      tokenizer: Uint8Array,
      tokenizerConfigJson: string,
      encoderConfigJson: string,
      rlAgentConfigJson: string,
    ): WasmAgent;
  };
  init_panic_hook: () => void;
};

let wasmReady: Promise<WasmExports> | null = null;

function loadWasm(): Promise<WasmExports> {
  if (!wasmReady) {
    wasmReady = (async () => {
      const mod = (await import("../wasm-pkg/laya.js")) as unknown as WasmExports;
      await mod.default();
      mod.init_panic_hook();
      return mod;
    })();
  }
  return wasmReady;
}

const CACHE_NAME = "laya-rs-checkpoints-v1";

export interface DownloadProgress {
  file: string;
  fileIndex: number;
  fileCount: number;
  /** Bytes loaded for the current file; total may be unknown (`null`) if the
   * server didn't send `content-length`. */
  loaded: number;
  total: number | null;
}

async function fetchWithProgress(url: string, onProgress: (loaded: number, total: number | null) => void): Promise<Uint8Array> {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(url);
  if (cached) {
    const buf = await cached.arrayBuffer();
    onProgress(buf.byteLength, buf.byteLength);
    return new Uint8Array(buf);
  }

  const resp = await fetch(url);
  if (!resp.ok || !resp.body) {
    throw new Error(`fetching ${url}: HTTP ${resp.status}`);
  }
  const totalHeader = resp.headers.get("content-length");
  const total = totalHeader ? Number(totalHeader) : null;

  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded, total);
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }

  // Best-effort: cache writes can fail (private browsing, quota) — the model
  // still loaded, so don't fail the whole load over it.
  try {
    await cache.put(url, new Response(bytes, { headers: { "content-type": "application/octet-stream" } }));
  } catch {
    /* ignore */
  }
  return bytes;
}

export interface LoadedModel {
  model: WasmAgent;
  def: ModelDef;
}

function textFile(bytes: Record<string, Uint8Array>, name: string): string {
  return new TextDecoder().decode(bytes[name]);
}

function loadFromFiles(bytes: Record<string, Uint8Array>, wasm: WasmExports): WasmAgent {
  return wasm.WasmAgent.load(
    bytes["model.safetensors"],
    bytes["tokenizer/tokenizer.json"],
    textFile(bytes, "tokenizer/tokenizer_config.json"),
    textFile(bytes, "encoder/config.json"),
    textFile(bytes, "rl_agent_config.json"),
  );
}

/** A local mirror of one or more checkpoints, picked once as a single parent
 * directory: `<root>/<hf-repo-basename>/model.safetensors`,
 * `.../tokenizer/tokenizer.json`, etc — one subdirectory per checkpoint,
 * named after the last segment of its Hugging Face repo id
 * (`convaiinnovations/laya-typed-decisions` -> `laya-typed-decisions`).
 *
 * Held as a [File System Access API](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API)
 * directory handle rather than an uploaded file list: picking it never
 * enumerates the whole tree, and the handle itself is what's remembered
 * across a reload — see [`ensureLocalLibraryResumeAttempted`]. */
export interface LocalLibrary {
  rootName: string;
  handle: FileSystemDirectoryHandle;
  checkpointNames: string[];
}

function hfRepoBasename(hfRepo: string): string {
  return hfRepo.split("/").pop() ?? hfRepo;
}

/** Which of `models` (by Hugging Face repo basename) are present in a
 * local library, in `models` order. */
export function availableLocally(library: LocalLibrary, models: ModelDef[]): ModelDef[] {
  const names = new Set(library.checkpointNames);
  return models.filter((m) => names.has(hfRepoBasename(m.hfRepo)));
}

/** Loads one checkpoint from a local library directory — no network
 * involved. Expects the same five files a Hugging Face checkpoint has (see
 * [`CHECKPOINT_FILES`]). */
export async function loadModelFromHandle(
  def: ModelDef,
  library: LocalLibrary,
  onProgress?: (p: DownloadProgress) => void,
): Promise<LoadedModel> {
  const wasm = await loadWasm();
  const checkpointDir = await getSubdirectory(library.handle, hfRepoBasename(def.hfRepo));
  const bytes: Record<string, Uint8Array> = {};
  for (let i = 0; i < CHECKPOINT_FILES.length; i++) {
    const name = CHECKPOINT_FILES[i];
    const file = await readRelativeFile(checkpointDir, name);
    onProgress?.({ file: name, fileIndex: i, fileCount: CHECKPOINT_FILES.length, loaded: 0, total: file.size });
    bytes[name] = new Uint8Array(await file.arrayBuffer());
    onProgress?.({ file: name, fileIndex: i, fileCount: CHECKPOINT_FILES.length, loaded: file.size, total: file.size });
  }
  const model = loadFromFiles(bytes, wasm);
  return { model, def };
}

/** Downloads (or reuses cached) checkpoint files for `def` and loads them
 * into a fresh `WasmAgent`. Reports per-file byte progress via `onProgress`. */
export async function loadModel(def: ModelDef, onProgress?: (p: DownloadProgress) => void): Promise<LoadedModel> {
  const wasm = await loadWasm();
  const files: Record<string, Uint8Array> = {};
  for (let i = 0; i < CHECKPOINT_FILES.length; i++) {
    const file = CHECKPOINT_FILES[i];
    const url = hfFileUrl(def.hfRepo, file);
    files[file] = await fetchWithProgress(url, (loaded, total) =>
      onProgress?.({ file, fileIndex: i, fileCount: CHECKPOINT_FILES.length, loaded, total }),
    );
  }
  const model = loadFromFiles(files, wasm);
  return { model, def };
}

/** Clears every cached checkpoint file (used by a "clear cache" control). */
export async function clearModelCache(): Promise<void> {
  await caches.delete(CACHE_NAME);
}

export async function cachedModelKeys(): Promise<string[]> {
  if (!("caches" in window)) return [];
  const cache = await caches.open(CACHE_NAME);
  const keys = await cache.keys();
  return keys.map((r) => r.url);
}

/** Shared "which model is currently loaded" state, so every demo tab can
 * reuse one loaded model instead of re-downloading per tab. */
const [current, setCurrent] = createSignal<LoadedModel | null>(null);
export const currentModel = current;

export type ModelSource = "hf" | "local";

/** What's remembered across reloads: which source, which Hugging Face
 * model, and (since a local directory's file handles can't survive a
 * reload) which local root/checkpoint name to look for once the user picks
 * that folder again. */
interface PersistedState {
  source: ModelSource;
  hfModelKey?: string;
  localRootName?: string;
  localModelKey?: string;
}

const LS_KEY = "laya-rs:demo-model-v1";

function readPersisted(): PersistedState {
  if (typeof localStorage === "undefined") return { source: "hf" };
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return { source: "hf" };
    const parsed = JSON.parse(raw) as Partial<PersistedState>;
    return { source: parsed.source === "local" ? "local" : "hf", ...parsed };
  } catch {
    return { source: "hf" };
  }
}

function writePersisted(patch: Partial<PersistedState>) {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(LS_KEY, JSON.stringify({ ...readPersisted(), ...patch }));
  } catch {
    /* quota / private-mode — resuming after reload just won't work this time */
  }
}

/** What was loaded last session, for the "resume on refresh" prompt — read
 * once per `ModelPicker` mount (not reactive: it only matters right after a
 * page load, before anything's picked again this session). */
export const persistedState = readPersisted;

/** The source (Hugging Face vs. local directory) the last successful load
 * used, shared across every `ModelPicker` instance so switching demo tabs
 * keeps whichever one the user picked instead of resetting to "Hugging
 * Face" each time — and restored from localStorage on page load. */
const [preferredSource, setPreferredSource] = createSignal<ModelSource>(readPersisted().source);
export { preferredSource };

/** The local library directory the user picked, shared across every
 * `ModelPicker` instance — pick it once, then every demo tab's "local
 * directory" source lists the checkpoints found inside it, no re-picking
 * per tab or per model. */
const [localLibrary, setLocalLibrary] = createSignal<LocalLibrary | null>(null);
export { localLibrary };

/** Set once a remembered directory handle is found but its permission
 * wasn't silently granted — the user needs one click (a `requestPermission`
 * call needs a user gesture; the browser dialog itself is skipped since the
 * directory was already picked before). See [`grantPendingLibraryAccess`]. */
const [pendingLibrary, setPendingLibrary] = createSignal<LocalLibrary | null>(null);
export { pendingLibrary };

async function toLibrary(handle: FileSystemDirectoryHandle): Promise<LocalLibrary> {
  return { rootName: handle.name, handle, checkpointNames: await listSubdirectories(handle) };
}

export async function selectLocalLibrary(): Promise<LocalLibrary | null> {
  const handle = await pickDirectory();
  if (!handle) return null; // user dismissed the picker
  await rememberDirectory(handle);
  const library = await toLibrary(handle);
  setLocalLibrary(library);
  setPendingLibrary(null);
  setPreferredSource("local");
  writePersisted({ source: "local", localRootName: library.rootName });
  return library;
}

/** Tries to reopen the directory handle remembered from a previous session,
 * entirely without a file picker. If the browser silently still grants read
 * permission (common — Chromium persists it), the library becomes active
 * right away; otherwise it's parked in `pendingLibrary` for one
 * [`grantPendingLibraryAccess`] click. Safe to call repeatedly — only the
 * first call in a session does anything. */
let resumeAttempted = false;
export function ensureLocalLibraryResumeAttempted(): void {
  if (resumeAttempted || localLibrary()) return;
  resumeAttempted = true;
  void (async () => {
    const handle = await loadRememberedDirectory();
    if (!handle) return;
    const granted = (await queryReadPermission(handle)) === "granted";
    const library = granted
      ? await toLibrary(handle)
      : { rootName: handle.name, handle, checkpointNames: [] as string[] };
    if (granted) {
      setLocalLibrary(library);
      setPreferredSource("local");
    } else {
      setPendingLibrary(library);
    }
  })();
}

/** Grants (or denies) access to the directory found by
 * [`ensureLocalLibraryResumeAttempted`] — call from a click handler, since
 * `requestPermission` needs a user gesture. No file picker dialog appears;
 * it's a lightweight browser permission prompt for the already-known
 * directory. */
export async function grantPendingLibraryAccess(): Promise<LocalLibrary | null> {
  const pending = pendingLibrary();
  if (!pending) return null;
  const granted = (await requestReadPermission(pending.handle)) === "granted";
  if (!granted) return null;
  const library = await toLibrary(pending.handle);
  setLocalLibrary(library);
  setPendingLibrary(null);
  setPreferredSource("local");
  return library;
}

export async function selectModel(def: ModelDef, onProgress?: (p: DownloadProgress) => void): Promise<LoadedModel> {
  const existing = current();
  if (existing && existing.def.hfRepo === def.hfRepo) {
    setPreferredSource("hf");
    writePersisted({ source: "hf", hfModelKey: def.key });
    return existing;
  }
  existing?.model.free();
  const loaded = await loadModel(def, onProgress);
  setCurrent(loaded);
  setPreferredSource("hf");
  writePersisted({ source: "hf", hfModelKey: def.key });
  return loaded;
}

export async function selectLocalModel(def: ModelDef, onProgress?: (p: DownloadProgress) => void): Promise<LoadedModel> {
  const library = localLibrary();
  if (!library) throw new Error("no local library directory selected");
  if (!library.checkpointNames.includes(hfRepoBasename(def.hfRepo))) {
    throw new Error(`${def.label} not found in ${library.rootName}`);
  }
  writePersisted({ source: "local", localRootName: library.rootName, localModelKey: def.key });
  const existing = current();
  if (existing && existing.def.key === def.key && preferredSource() === "local") return existing;
  existing?.model.free();
  const loaded = await loadModelFromHandle(def, library, onProgress);
  setCurrent(loaded);
  setPreferredSource("local");
  return loaded;
}

export function unloadModel(): void {
  current()?.model.free();
  setCurrent(null);
}
