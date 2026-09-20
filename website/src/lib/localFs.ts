/** The File System Access API, narrowed to what the model-library picker
 * needs: pick a directory once, keep its handle (in IndexedDB, so it
 * survives a reload), list its immediate subdirectories, and read one file
 * out of one of them — without ever enumerating the whole tree. A laya
 * checkpoint mirror can hold several checkpoints' worth of files; this never
 * lists more than the handful actually needed. */

type PickerWindow = Window & {
  showDirectoryPicker(options?: { mode?: "read" | "readwrite"; id?: string }): Promise<FileSystemDirectoryHandle>;
};

/** The picker and permission calls are still ahead of the DOM typings in
 * some TypeScript releases, so they're reached through this intersection
 * rather than by merging into `lib.dom`. */
type PermissionHandle = FileSystemDirectoryHandle & {
  entries(): AsyncIterable<[string, FileSystemHandle]>;
  queryPermission(options: { mode: "read" | "readwrite" }): Promise<PermissionState>;
  requestPermission(options: { mode: "read" | "readwrite" }): Promise<PermissionState>;
};

export function fsAccessSupported(): boolean {
  return typeof window !== "undefined" && "showDirectoryPicker" in window;
}

export async function pickDirectory(): Promise<FileSystemDirectoryHandle | null> {
  try {
    return await (window as unknown as PickerWindow).showDirectoryPicker({ mode: "read", id: "laya-rs-models" });
  } catch (e) {
    // The user dismissing the picker is not a failure worth reporting.
    if (e instanceof DOMException && e.name === "AbortError") return null;
    throw e;
  }
}

export async function queryReadPermission(handle: FileSystemDirectoryHandle): Promise<PermissionState> {
  return (handle as PermissionHandle).queryPermission({ mode: "read" });
}

export async function requestReadPermission(handle: FileSystemDirectoryHandle): Promise<PermissionState> {
  return (handle as PermissionHandle).requestPermission({ mode: "read" });
}

export async function listSubdirectories(dir: FileSystemDirectoryHandle): Promise<string[]> {
  const names: string[] = [];
  for await (const [name, entry] of (dir as PermissionHandle).entries()) {
    if (entry.kind === "directory") names.push(name);
  }
  return names.sort((a, b) => a.localeCompare(b));
}

export async function getSubdirectory(dir: FileSystemDirectoryHandle, name: string): Promise<FileSystemDirectoryHandle> {
  return dir.getDirectoryHandle(name);
}

/** Reads one file by a `/`-separated path relative to `dir` (e.g.
 * `tokenizer/tokenizer.json`), without touching any of its siblings. */
export async function readRelativeFile(dir: FileSystemDirectoryHandle, relPath: string): Promise<File> {
  const segments = relPath.split("/");
  let cur = dir;
  for (let i = 0; i < segments.length - 1; i++) {
    cur = await cur.getDirectoryHandle(segments[i]);
  }
  const fileHandle = await cur.getFileHandle(segments[segments.length - 1]);
  return fileHandle.getFile();
}

/* -------------------------------------------------------------------- */
/* Remembering the picked directory's handle across reloads (IndexedDB —  */
/* handles aren't JSON-serializable, so localStorage can't hold them).    */
/* -------------------------------------------------------------------- */

const DB_NAME = "laya-rs-demo";
const STORE_NAME = "handles";
const HANDLE_KEY = "local-library";

function openDatabase(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === "undefined") return Promise.resolve(null);
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) db.createObjectStore(STORE_NAME);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("could not open IndexedDB"));
  });
}

async function withStore<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T | undefined> {
  const db = await openDatabase();
  if (!db) return undefined;
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, mode);
    const store = tx.objectStore(STORE_NAME);
    const request = run(store);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB request failed"));
    tx.oncomplete = () => db.close();
    tx.onerror = () => {
      db.close();
      reject(tx.error ?? new Error("IndexedDB transaction failed"));
    };
  });
}

export async function rememberDirectory(handle: FileSystemDirectoryHandle): Promise<void> {
  try {
    await withStore("readwrite", (store) => store.put(handle, HANDLE_KEY));
  } catch {
    /* best effort only — resuming after reload just won't work this time */
  }
}

export async function forgetRememberedDirectory(): Promise<void> {
  try {
    await withStore("readwrite", (store) => store.delete(HANDLE_KEY));
  } catch {
    /* ignore */
  }
}

export async function loadRememberedDirectory(): Promise<FileSystemDirectoryHandle | null> {
  try {
    const handle = await withStore<FileSystemDirectoryHandle | undefined>("readonly", (store) => store.get(HANDLE_KEY));
    return handle ?? null;
  } catch {
    return null;
  }
}
