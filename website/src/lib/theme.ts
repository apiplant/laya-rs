/**
 * Light and dark: dark by default, with a `light` class on `<html>` swapping
 * the palette.
 *
 * The choice is persisted; without one, the system preference applies. The
 * class is also set by an inline script in index.html so the first paint uses
 * the correct theme.
 */

import { createSignal } from "solid-js";

export type Theme = "dark" | "light";

const STORAGE_KEY = "laya-rs-theme";

function systemTheme(): Theme {
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function stored(): Theme | null {
  const value = localStorage.getItem(STORAGE_KEY);
  return value === "dark" || value === "light" ? value : null;
}

const [theme, setThemeSignal] = createSignal<Theme>(stored() ?? systemTheme());

function apply(next: Theme) {
  const root = document.documentElement;
  root.classList.toggle("light", next === "light");
  root.style.colorScheme = next;
}

apply(theme());

export function setTheme(next: Theme) {
  setThemeSignal(next);
  localStorage.setItem(STORAGE_KEY, next);
  apply(next);
}

export function toggleTheme() {
  setTheme(theme() === "dark" ? "light" : "dark");
}

/** Follow the system preference until the reader chooses explicitly. */
window.matchMedia?.("(prefers-color-scheme: light)").addEventListener("change", () => {
  if (!stored())
    setThemeSignal(() => {
      const next = systemTheme();
      apply(next);
      return next;
    });
});

export { theme };
