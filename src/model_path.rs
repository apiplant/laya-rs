//! Shared checkpoint-directory resolution for the CLI.
//!
//! Two checkpoint layouts are recognized, matching the website demo's model
//! picker (see `website/src/lib/laya.ts`'s `LocalLibrary` and
//! `localDirCandidates`): standalone per-variant Hugging Face repos (one
//! full checkpoint each, `convaiinnovations/laya-<variant>`) and the
//! `convaiinnovations/laya` hub repo — or a local clone of it, e.g. `~/laya`
//! — which bundles every variant as a same-named subfolder.
//!
//! Resolution order for a variant: an explicit `--model`/`LAYA_MODEL` path,
//! if given, wins outright and is used as-is (a self-contained checkpoint
//! directory, no variant lookup at all — this is the "individual model
//! downloaded from Hugging Face and pointed at directly" case). Otherwise,
//! if a family root is given (`--models-root`/`LAYA_MODELS_ROOT` — there is
//! no default; without one this step is skipped entirely), the variant's
//! subfolder under it is used when present. Failing both, the variant's own
//! cached standalone download is used (`$XDG_CACHE_HOME/laya-rs`, default
//! `~/.cache/laya-rs`), fetching it there first if needed (see
//! `crate::download`).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// One checkpoint the CLI can target: a short key, the Hugging Face repo it
/// downloads from standalone, and the subfolder name it has inside the
/// `convaiinnovations/laya` hub repo (or a local clone of it).
pub struct VariantDef {
    pub key: &'static str,
    pub hf_repo: &'static str,
    pub subfolder: &'static str,
}

/// Mirrors the website demo's `CHECKPOINT_MODELS` (website/src/lib/models.ts).
pub const VARIANTS: &[VariantDef] = &[
    VariantDef { key: "typed-decisions", hf_repo: "convaiinnovations/laya-typed-decisions", subfolder: "typed-decisions" },
    VariantDef { key: "multilingual", hf_repo: "convaiinnovations/laya-multilingual", subfolder: "multilingual" },
];

pub fn find(key: &str) -> Result<&'static VariantDef> {
    VARIANTS.iter().find(|v| v.key == key).with_context(|| format!("unknown model variant {key:?}"))
}

fn hf_repo_basename(hf_repo: &str) -> &str {
    hf_repo.rsplit('/').next().unwrap_or(hf_repo)
}

/// Subdirectory names that would hold `variant`'s files directly under a
/// family root, in preference order: the standalone repo's own basename
/// first, then the hub repo's subfolder name (same order and reasoning as
/// the website's `localDirCandidates`).
fn family_root_candidates(variant: &VariantDef) -> [&str; 2] {
    [hf_repo_basename(variant.hf_repo), variant.subfolder]
}

/// `variant`'s directory under `family_root`, if one of its recognized
/// subfolder names exists there and looks like a checkpoint (has a
/// `model.safetensors`).
fn find_in_family_root(family_root: &Path, variant: &VariantDef) -> Option<PathBuf> {
    family_root_candidates(variant).into_iter().map(|name| family_root.join(name)).find(|dir| dir.join("model.safetensors").is_file())
}

/// Resolves the checkpoint directory to load for one variant. `explicit` is
/// whatever the user passed via `--model`/`LAYA_MODEL`, if anything;
/// `requested_variant` is the variant key explicitly requested, if any, else
/// `default_variant_key` is used; `family_root` is an optional
/// `convaiinnovations/laya`-shaped directory to look in first.
pub fn resolve(
    default_variant_key: &str,
    explicit: Option<PathBuf>,
    requested_variant: Option<&str>,
    family_root: Option<&Path>,
) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path);
    }

    let variant_key = requested_variant.unwrap_or(default_variant_key);
    let variant = find(variant_key)?;

    if let Some(root) = family_root {
        if let Some(dir) = find_in_family_root(root, variant) {
            return Ok(dir);
        }
    }

    crate::download::download_variant(variant)
}
