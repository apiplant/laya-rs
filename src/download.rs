//! Downloads a standalone checkpoint from Hugging Face into the shared model cache.
//!
//! Cache location: `$XDG_CACHE_HOME/laya-rs/<repo basename>`, falling back to
//! `~/.cache/laya-rs/<repo basename>` when `$XDG_CACHE_HOME` is unset. Files are
//! fetched to a `.part` sibling and renamed into place once complete, so a killed
//! download never leaves a checkpoint that looks done.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::model_path::VariantDef;

/// Files every checkpoint needs in order to load (see `RLAgent::load`), also
/// mirrored by the website demo's `CHECKPOINT_FILES` (website/src/lib/models.ts).
const REQUIRED_FILES: &[&str] =
    &["rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "model.safetensors"];

/// `$XDG_CACHE_HOME/laya-rs`, defaulting to `~/.cache/laya-rs`.
pub fn cache_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("laya-rs"))
}

/// The checkpoint directory name for a variant: the repo name portion of
/// its `hf_repo` (e.g. `convaiinnovations/laya-multilingual` -> `laya-multilingual`).
fn checkpoint_dir_name(variant: &VariantDef) -> &str {
    variant.hf_repo.rsplit('/').next().unwrap_or(variant.hf_repo)
}

/// Where `download_variant` puts (or would put) a variant's checkpoint.
pub fn variant_cache_dir(variant: &VariantDef) -> Option<PathBuf> {
    Some(cache_dir()?.join(checkpoint_dir_name(variant)))
}

fn fetch_to_file(url: &str, dest: &Path) -> Result<()> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let part = dest.with_extension(match dest.extension() {
        Some(ext) => format!("{}.part", ext.to_string_lossy()),
        None => "part".to_string(),
    });

    let resp = ureq::get(url).call().with_context(|| format!("downloading {url}"))?;
    let len: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());

    eprint!("  {} ...", dest.file_name().unwrap_or_default().to_string_lossy());
    std::io::stderr().flush().ok();

    let mut file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut reader = resp.into_reader();
    let mut buf = [0u8; 1 << 16];
    let mut written: u64 = 0;
    loop {
        let n = reader.read(&mut buf).with_context(|| format!("reading body for {url}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).with_context(|| format!("writing {}", part.display()))?;
        written += n as u64;
    }
    drop(file);

    if let Some(len) = len {
        if written != len {
            let _ = std::fs::remove_file(&part);
            bail!("short read for {url}: got {written} bytes, expected {len}");
        }
    }
    std::fs::rename(&part, dest).with_context(|| format!("renaming {} to {}", part.display(), dest.display()))?;
    eprintln!(" done ({written} bytes)");
    Ok(())
}

/// Downloads `variant`'s checkpoint from its Hugging Face repo into the
/// laya-rs cache directory, skipping files already present, and returns
/// the checkpoint directory. Requires network access; any failure (offline,
/// 404, disk error, ...) is returned as an error and nothing partial is left
/// looking complete.
pub fn download_variant(variant: &VariantDef) -> Result<PathBuf> {
    let dir = variant_cache_dir(variant).context("no cache directory available (set $HOME or $XDG_CACHE_HOME)")?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    let mut announced = false;
    for file in REQUIRED_FILES {
        let dest = dir.join(file);
        if dest.is_file() {
            continue;
        }
        if !announced {
            eprintln!("Downloading {} from https://huggingface.co/{} into {}", variant.key, variant.hf_repo, dir.display());
            announced = true;
        }
        if std::env::var_os("LAYA_OFFLINE").is_some() {
            bail!(
                "{} is missing from {} and LAYA_OFFLINE is set; unset it to allow downloading, or pass --model",
                file,
                dir.display()
            );
        }
        let url = format!("https://huggingface.co/{}/resolve/main/{}", variant.hf_repo, file);
        fetch_to_file(&url, &dest)?;
    }
    Ok(dir)
}
