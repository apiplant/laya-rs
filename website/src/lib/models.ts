/** laya checkpoints the browser demo can download from Hugging Face,
 * mirroring the two checkpoints `router::route` picks between (see
 * `src/router.rs`) — the demo's model picker chooses the checkpoint
 * directly instead of routing on the state text. */

export const CHECKPOINT_FILES = [
  "rl_agent_config.json",
  "encoder/config.json",
  "tokenizer/tokenizer.json",
  "tokenizer/tokenizer_config.json",
  "model.safetensors",
] as const;

export interface ModelDef {
  key: string;
  label: string;
  /** The standalone repo this checkpoint downloads from over HTTP. */
  hfRepo: string;
  /** `convaiinnovations/laya` is the hub for the whole family: the English
   * checkpoint sits at that repo's root, and `multilingual`/`typed-decisions`
   * are subfolders of it. A local checkpoint family clone (e.g. `~/laya`,
   * the CLI's own `models_root` convention) mirrors that same layout, so
   * this name doubles as the local-directory subfolder to look for,
   * alongside the standalone repo's own basename — see
   * `lib/laya.ts`'s `localDirCandidates`. */
  subfolder: string;
  /** Rough `model.safetensors` size, for the picker — not fetched ahead of time. */
  approxSizeMb: number;
  description: string;
}

export const CHECKPOINT_MODELS: ModelDef[] = [
  {
    key: "typed-decisions",
    label: "laya-typed-decisions",
    hfRepo: "convaiinnovations/laya-typed-decisions",
    subfolder: "typed-decisions",
    approxSizeMb: 843,
    description: "English, ModernBERT-large backbone (421M params) — the checkpoint tuned for typed choice/score/noul decisions. Default.",
  },
  {
    key: "multilingual",
    label: "laya-multilingual",
    hfRepo: "convaiinnovations/laya-multilingual",
    subfolder: "multilingual",
    approxSizeMb: 644,
    description: "Multilingual, mmBERT-base backbone (322M params) — for state text outside English or the Latin alphabet.",
  },
];

export function hfFileUrl(repo: string, file: string): string {
  return `https://huggingface.co/${repo}/resolve/main/${file}`;
}
