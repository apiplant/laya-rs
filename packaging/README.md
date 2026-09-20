# Packaging

One package, `laya-rs`, carrying the `laya` binary, plus the `laya-rs`
library published to crates.io. The `packages`, `homebrew`, `apt` and
`pacman` jobs in
[`.github/workflows/release.yml`](../.github/workflows/release.yml) substitute
the `@VERSION@`, `@SHA_*@` and `@ARCH@` placeholders with the tag's version and
the checksums of the archives that release just built, then publish the
result. None of the definitions compiles the project from source — they all
install the binary the `binaries`/`flash-attn-binary` jobs already produced,
which is what keeps the packages and the release byte-identical. The
`crates-io` job is the exception: `cargo publish` builds from source, because
that's what publishing a library to crates.io means.

| File | Publishes to |
| --- | --- |
| `homebrew/laya-rs.rb` | `apiplant/homebrew-tap`, as `Formula/laya-rs.rb` |
| `pacman/PKGBUILD` | the release itself, as a `.pkg.tar.zst` asset, and `apiplant/pacman` |
| `pacman/PKGBUILD-flash-attn` | the release itself, as a `.pkg.tar.zst` asset, and `apiplant/pacman` |
| `debian/control` | the release itself, as a `.deb` asset |
| `debian/control-flash-attn` | the release itself, as a `.deb` asset |
| `apt/apt-ftparchive.conf` | `apiplant/apt`, served at `apt.apiplant.com` |
| (n/a — `cargo publish`) | crates.io, as the `laya-rs` crate |

This repository reuses the shared `apiplant` repositories that already serve
`nvidia-smi-live`, `portward`, `megadl-rs` and `gliner-rs` — `apiplant/homebrew-tap`,
`apiplant/pacman` (served at `apiplant.github.io/pacman`) and `apiplant/apt`
(served at `apt.apiplant.com`) — rather than standing up new ones. The same
repository secrets those releases use (`HOMEBREW_TAP_TOKEN`,
`PACMAN_REPO_TOKEN` + `PACMAN_GPG_PRIVATE_KEY`/`PACMAN_GPG_PASSPHRASE`,
`APT_REPO_TOKEN` + `APT_GPG_PRIVATE_KEY`/`APT_GPG_PASSPHRASE`) work here
unchanged; if this repository doesn't have them set yet, copy them over from
one of the others as repository secrets. Publishing to crates.io needs its
own `CARGO_REGISTRY_TOKEN` secret (an API token from
https://crates.io/settings/tokens, scoped to `publish-update` on the
`laya-rs` crate). A publish job whose credential is absent is skipped
rather than failing the release, so a fork — or this repository before the
secrets are copied — still gets a clean release.

## Platform matrix

| Platform | How it ships |
| --- | --- |
| macOS (Apple Silicon, `aarch64-apple-darwin`) | archive + Homebrew formula |
| Linux x86_64 (`x86_64-unknown-linux-gnu`) | archive, `.deb` + apt repo, Arch package + pacman repo, Homebrew formula |
| Linux x86_64, CUDA + flash-attn (`x86_64-unknown-linux-gnu` + `--features flash-attn`) | archive, `.deb` + apt repo, Arch package + pacman repo |
| Linux aarch64 (`aarch64-unknown-linux-gnu`) | archive, `.deb` + apt repo, Homebrew formula |
| Any platform Cargo runs on | `cargo install laya-rs`, via crates.io |

The Arch package is x86_64 only: it would need an arm runner or emulation to
build, and the Arch repository has no aarch64 audience. The `.deb`, the apt
repo and the plain archive still cover Linux arm64.

There's no separate plain-`cuda` build: laya's own recommended GPU path is
`--features flash-attn` (which also pulls in `cuda`), so that's the only GPU
archive/package shipped — anyone who wants CUDA without flash-attn builds
from source with `--features cuda`. The `flash-attn-binary` job compiles it
in an `nvidia/cuda:*-devel-ubuntu22.04` container — that gives `nvcc` and the
CUDA headers/stub libraries the build needs (`candle-core`'s `cuda` feature
links `cudarc` against them, and `candle-flash-attn` compiles its own CUDA
kernels against NVIDIA's cutlass headers on first build), without needing an
actual GPU on the runner. It ships as `laya-rs-flash-attn`, a separate `.deb`
(`packaging/debian/control-flash-attn`) and Arch package
(`packaging/pacman/PKGBUILD-flash-attn`), published to the same apt and
pacman repositories as the CPU build. Neither package can express "needs a
host NVIDIA driver compatible with this CUDA toolkit version" precisely — the
`.deb` depends on `libcuda1` and the Arch package on `nvidia-utils`, which get
the runtime library installed but don't check the driver actually supports
this CUDA toolkit version, so a user installing it still needs to know that
and have it. Both packages `Conflicts`/`conflicts` and `Provides`/`provides`
`laya-rs`, since they install the same `laya` binary name and only one build
can be on a system at a time. Not published to Homebrew: the formula has no
mechanism for a CUDA-variant formula. There is no aarch64 or macOS GPU
build — no CUDA on Apple Silicon, and no arm64 CUDA audience to justify the
extra job.

The order is: build every archive, build the distro packages from those
archives, publish the release, then publish to crates.io, Homebrew, apt and
pacman — the formula, apt pool, PKGBUILD and crates.io entry all reference (or
are built from) that same tagged commit, and the GitHub release should exist
first so the archives it links stay in sync.

## Using the packages

macOS (Apple Silicon) and Linux, via Homebrew:

```bash
brew tap apiplant/tap
brew install apiplant/tap/laya-rs
```

Arch Linux, via the signed pacman repository at `apiplant.github.io/pacman`
(one-time setup, then `pacman -Sy`/`-Syu` picks up new releases):

```bash
curl -sSfL https://apiplant.github.io/pacman/apiplant.gpg -o /tmp/apiplant.gpg
keyid=$(gpg --show-keys --with-colons /tmp/apiplant.gpg | awk -F: '/^pub:/ { print $5; exit }') && sudo pacman-key --add /tmp/apiplant.gpg && sudo pacman-key --finger "$keyid" && sudo pacman-key --lsign-key "$keyid"
printf '\n[apiplant]\nSigLevel = Required DatabaseOptional\nServer = https://apiplant.github.io/pacman/$arch\n' | sudo tee -a /etc/pacman.conf > /dev/null
sudo pacman -Sy laya-rs
```

Debian/Ubuntu, via the signed apt repository at `apt.apiplant.com` (one-time
setup, then `apt upgrade` picks up new releases):

```bash
curl -sSfL https://apt.apiplant.com/apiplant-archive-keyring.gpg | sudo tee /usr/share/keyrings/apiplant.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/apiplant.gpg] https://apt.apiplant.com stable main" | sudo tee /etc/apt/sources.list.d/apiplant.list > /dev/null
sudo apt update && sudo apt install laya-rs
```

Or install a single `.deb`/`.pkg.tar.zst` release asset directly without
adding a repository:

```bash
sudo dpkg -i laya-rs_*_amd64.deb
sudo pacman -U laya-rs-*-x86_64.pkg.tar.zst
```

Or just download and unpack the archive for your platform from the release
page — the `laya` binary is static enough to run from anywhere, no
installation required. On Linux x86_64 with an NVIDIA GPU, download the
`-flash-attn` archive instead
(`laya-rs-flash-attn-<tag>-x86_64-unknown-linux-gnu.tar.gz`) for
CUDA + flash-attn-accelerated inference; it needs a host driver compatible
with the CUDA toolkit it was built against.

As a Rust library, or to build the CLI from source, via crates.io:

```bash
cargo add laya-rs         # as a library dependency
cargo install laya-rs     # for the laya binary
cargo install laya-rs --features flash-attn  # with CUDA + flash-attn support
```
