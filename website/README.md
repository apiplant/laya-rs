# laya-rs website

The marketing and documentation site for [laya-rs](https://github.com/apiplant/laya-rs).
Solid 2 RC + Tailwind v4 + Vite, static build, deployed to Cloudflare Pages.

```bash
pnpm install
pnpm dev       # http://127.0.0.1:5276
pnpm build     # → dist/
pnpm check     # types only
pnpm build:wasm  # regenerate src/wasm-pkg/ (needs Rust + wasm-pack)
```

## The wasm package is committed

The `/demo` page runs laya-rs itself in the browser, so the site depends on
`src/wasm-pkg/` — the `wasm-pack` output for the crate one directory up. That
directory is **committed** rather than generated during the build: the deploy
environment has Node and nothing else, and asking it for a Rust toolchain
would mean installing rustup and compiling candle on every deploy.

So `pnpm build` never runs `wasm-pack`; it only checks the package is there.
After changing anything under `../src/`, run `pnpm build:wasm` and commit the
result, or the site keeps serving the previous build of the crate.

Note that the wasm build uses `tokenizers`' `unstable_wasm` feature (a
pure-Rust `fancy-regex` backend) for every target, not just wasm32, since
Cargo can't resolve different features for the same dependency across
targets on stable — see the comment in `../Cargo.toml`. The native CLI and
the browser demo therefore share the same tokenization backend.

## The version is not copied here

The install section names the version, and the binary stamps the same
string — so `vite.config.ts` reads it from `../Cargo.toml` (`[package]
version`) at build time and injects it as `__VERSION__`. There is no copy in
the site to keep in sync.

## Documentation pages

`src/components/docs/` holds one page per topic (`Overview`, `Library`,
`Cli`, `Training`), a shared sidebar (`DocsLayout.tsx`), and shared prose
primitives (`Prose.tsx` — headings, paragraphs, code blocks, flag tables).
There is no MDX pipeline: each page is hand-written TSX kept in sync with
`../README.md` by hand.

## The playground

Unlike a schema-conversion demo, laya's own question schema (`choice` /
`score` / `noul`, see `src/components/demo/QuestionBuilder.tsx`) is exactly
what the model consumes — `WasmAgent::ask`'s `questions_json` parameter takes
the builder's output directly, with no intermediate format. The builder
keeps richer in-progress row state (half-typed options, empty new levels)
than the serialized shape, and only serializes on each change via a
`createEffect`.

## Deploying

A static SPA: `dist/` is assets-only, and `wrangler.jsonc` sends unknown paths
to `index.html` (`not_found_handling: single-page-application`) so deep links
resolve on a cold load.

```bash
npx wrangler pages deploy dist --project-name laya-rs-website
```

`index.html`, `public/robots.txt` and `public/sitemap.xml` name the default
`laya-rs.apiplant.com` domain; update all three if the site moves.
