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
  hfRepo: string;
  /** Rough `model.safetensors` size, for the picker — not fetched ahead of time. */
  approxSizeMb: number;
  description: string;
}

export const CHECKPOINT_MODELS: ModelDef[] = [
  {
    key: "typed-decisions",
    label: "laya-typed-decisions",
    hfRepo: "convaiinnovations/laya-typed-decisions",
    approxSizeMb: 843,
    description: "English, ModernBERT-large backbone (421M params) — the checkpoint tuned for typed choice/score/noul decisions. Default.",
  },
  {
    key: "multilingual",
    label: "laya-multilingual",
    hfRepo: "convaiinnovations/laya-multilingual",
    approxSizeMb: 644,
    description: "Multilingual, mmBERT-base backbone (322M params) — for state text outside English or the Latin alphabet.",
  },
];

export function hfFileUrl(repo: string, file: string): string {
  return `https://huggingface.co/${repo}/resolve/main/${file}`;
}
