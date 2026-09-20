/**
 * Fails fast, with an actionable message, when the generated wasm package is
 * missing — otherwise Vite reports it as an unresolved import of
 * `../wasm-pkg/laya.js`, which says nothing about what to do.
 *
 * `src/wasm-pkg/` is committed precisely so a build needs nothing but Node
 * (the deploy environment has no Rust toolchain), so a missing one means a
 * bad checkout rather than a normal first-run state.
 */

import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const required = ["laya.js", "laya_bg.wasm"];
const dir = new URL("../src/wasm-pkg/", import.meta.url);

const missing = required.filter((name) => !existsSync(new URL(name, dir)));
if (missing.length > 0) {
  console.error(
    `\nMissing ${missing.join(", ")} in ${fileURLToPath(dir)}\n\n` +
      "The wasm package is committed to the repo. Regenerate it with:\n" +
      "  pnpm build:wasm      (needs Rust + wasm-pack)\n",
  );
  process.exit(1);
}
