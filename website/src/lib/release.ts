/**
 * The published release and its assets.
 *
 * `release.yml` builds one archive per target and names it
 * `laya-rs-<tag>-<target>.tar.gz` (plus a separate CUDA + flash-attn
 * `laya-rs-flash-attn-<tag>-x86_64-unknown-linux-gnu.tar.gz`), so a direct
 * link can be assembled here rather than sending every reader to the
 * releases page to hunt for their own. The version comes from the crate
 * manifest at build time, which means a site deployed from a commit
 * predating the tag would link at an asset that does not exist yet — deploy
 * the site after the release, not before.
 */

import { GITHUB_URL } from "./links";

export const VERSION = __VERSION__;
/** Git tags are `v`-prefixed, and the tag is part of every asset name. */
export const TAG = `v${VERSION}`;

export const RELEASES_URL = `${GITHUB_URL}/releases`;
export const LATEST_RELEASE_URL = `${RELEASES_URL}/tag/${TAG}`;

export interface Platform {
  /** Rust target triple, as it appears in the asset name. */
  target: string;
  label: string;
  /** For the download button, which cannot wrap and sits in a narrow column. */
  short: string;
  flashAttn?: boolean;
}

/* Exactly the targets release.yml builds: three CPU archives, plus a
   CUDA + flash-attn Linux x86_64 archive built as a separate job. */
export const PLATFORMS: Platform[] = [
  { target: "x86_64-unknown-linux-gnu", label: "Linux · x86_64", short: "Linux x86_64" },
  { target: "aarch64-unknown-linux-gnu", label: "Linux · aarch64", short: "Linux aarch64" },
  { target: "aarch64-apple-darwin", label: "macOS · Apple Silicon", short: "macOS (Apple Silicon)" },
];

export function assetName(platform: Platform): string {
  const prefix = platform.flashAttn ? "laya-rs-flash-attn" : "laya-rs";
  return `${prefix}-${TAG}-${platform.target}.tar.gz`;
}

export function downloadUrl(platform: Platform): string {
  return `${RELEASES_URL}/download/${TAG}/${assetName(platform)}`;
}

export const FLASH_ATTN_PLATFORM: Platform = {
  target: "x86_64-unknown-linux-gnu",
  label: "Linux · x86_64 (CUDA + flash-attn)",
  short: "Linux x86_64, CUDA + flash-attn",
  flashAttn: true,
};
