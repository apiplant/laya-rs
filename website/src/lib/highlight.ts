import Prism from "prismjs";
import "prismjs/components/prism-rust";
import "prismjs/components/prism-bash";
import "prismjs/components/prism-json";

export type HighlightLang = "rust" | "bash" | "json" | "console" | "text";

/** console/text aren't real Prism grammars: fall back to bash for console
 *  (it's mostly shell prompts and output) and skip highlighting for text. */
function grammarFor(lang: HighlightLang) {
  if (lang === "console") return { grammar: Prism.languages.bash, name: "bash" };
  if (lang === "rust" || lang === "bash" || lang === "json") {
    return { grammar: Prism.languages[lang], name: lang };
  }
  return null;
}

/** Returns Prism-tokenized HTML (`<span class="token ...">`), or the
 *  escaped source unchanged when the language has no grammar. */
export function highlight(code: string, lang: HighlightLang): string {
  const found = grammarFor(lang);
  if (!found) return escapeHtml(code);
  return Prism.highlight(code, found.grammar, found.name);
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]!);
}
