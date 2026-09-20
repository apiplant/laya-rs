/* tslint:disable */
/* eslint-disable */

export class WasmAgent {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Answers a batch of typed questions against one `state` (arbitrary
     * JSON, or a plain string body).
     *
     * `questions_json` is a JSON object of `{qid: {type, instructions,
     * criteria}}` (see [`crate::batching::RawQuestion`]). Returns JSON
     * `{qid: {type, ...}}`, one answer per question, in the same shape
     * `laya jev` writes.
     */
    ask(state_json: string, questions_json: string): string;
    /**
     * Loads a checkpoint from its five file contents, on CPU in F32 (the
     * only combination that makes sense in a browser tab).
     *
     * `weights` is `model.safetensors`; `tokenizer` is
     * `tokenizer/tokenizer.json`; `tokenizer_config_json` is
     * `tokenizer/tokenizer_config.json`; `encoder_config_json` is
     * `encoder/config.json`; `rl_agent_config_json` is
     * `rl_agent_config.json`.
     */
    static load(weights: Uint8Array, tokenizer: Uint8Array, tokenizer_config_json: string, encoder_config_json: string, rl_agent_config_json: string): WasmAgent;
}

/**
 * Call once from JS before anything else, to get readable panic messages
 * (from candle shape mismatches etc.) in the browser console.
 */
export function init_panic_hook(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmagent_free: (a: number, b: number) => void;
    readonly init_panic_hook: () => void;
    readonly wasmagent_ask: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly wasmagent_load: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number, number];
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
