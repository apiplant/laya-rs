/** Shared presentational pieces. */

import { Show, omit, type ParentProps } from "solid-js";
import type { JSX } from "@solidjs/web";
import { theme, toggleTheme } from "../lib/theme";

type Variant = "primary" | "secondary" | "ghost";

const VARIANTS: Record<Variant, string> = {
  primary: "bg-accent text-on-accent hover:bg-accent-dim active:bg-accent-dim font-semibold",
  secondary: "bg-surface-2 text-ink border border-line hover:border-line-strong hover:bg-surface-3",
  ghost: "text-muted hover:text-ink hover:bg-surface-2",
};

const SIZES = {
  sm: "px-2.5 py-1 text-xs",
  md: "px-3.5 py-2 text-sm",
  lg: "px-5 py-2.5 text-[0.9375rem]",
};

function buttonClass(variant: Variant | undefined, size: keyof typeof SIZES | undefined, extra?: string) {
  return [
    "inline-flex items-center justify-center gap-2 rounded-lg whitespace-nowrap transition-colors duration-100",
    "disabled:opacity-40 disabled:pointer-events-none",
    SIZES[size ?? "md"],
    VARIANTS[variant ?? "secondary"],
    extra ?? "",
  ].join(" ");
}

export function Button(
  props: ParentProps<
    { variant?: Variant; size?: keyof typeof SIZES; class?: string } & JSX.ButtonHTMLAttributes<HTMLButtonElement>
  >,
) {
  const rest = omit(props, "variant", "size", "class", "children");
  return (
    <button {...rest} class={buttonClass(props.variant, props.size, props.class)}>
      {props.children}
    </button>
  );
}

/** The same shape as `Button`, for links that act like one. */
export function LinkButton(
  props: ParentProps<
    { variant?: Variant; size?: keyof typeof SIZES; class?: string; href: string } & JSX.AnchorHTMLAttributes<HTMLAnchorElement>
  >,
) {
  const rest = omit(props, "variant", "size", "class", "children", "href");
  const external = () => /^https?:/.test(props.href);

  return (
    <Show
      when={!external()}
      fallback={
        <a
          {...rest}
          href={props.href}
          target="_blank"
          rel="noreferrer noopener"
          class={buttonClass(props.variant, props.size, props.class)}
        >
          {props.children}
        </a>
      }
    >
      <a {...rest} href={props.href} class={buttonClass(props.variant, props.size, props.class)}>
        {props.children}
      </a>
    </Show>
  );
}

export function Badge(props: ParentProps<{ tone?: "neutral" | "accent" | "warn"; class?: string }>) {
  const tones = {
    neutral: "bg-surface-3 text-muted border-line",
    accent: "bg-accent-soft text-accent border-accent-line",
    warn: "bg-surface-3 text-warn border-line-strong",
  };
  return (
    <span
      class={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[0.6875rem] font-medium leading-5 ${
        tones[props.tone ?? "neutral"]
      } ${props.class ?? ""}`}
    >
      {props.children}
    </span>
  );
}

export function Mono(props: ParentProps<{ class?: string }>) {
  return (
    <code class={`font-mono text-[0.9em] text-muted ${props.class ?? ""}`}>{props.children}</code>
  );
}

/** The mark: a checkmark, for a typed question answered — laya's whole job. */
export function LayaMark(props: { class?: string }) {
  return (
    <svg
      viewBox="0 0 32 32"
      aria-hidden="true"
      class={`shrink-0 ${props.class ?? "h-6 w-6"}`}
      fill="none"
    >
      <rect x="1.5" y="1.5" width="29" height="29" rx="7" class="stroke-line" stroke-width="1.5" />
      <path
        d="m9 17 4.5 4.5L23 11.5"
        stroke="currentColor"
        stroke-width="2.4"
        stroke-linecap="round"
        stroke-linejoin="round"
        class="text-accent"
      />
    </svg>
  );
}

/** `laya-rs`, two-tone. */
export function Wordmark(props: { class?: string }) {
  return (
    <span class={`inline-flex min-w-0 items-center gap-2 ${props.class ?? ""}`}>
      <LayaMark class="h-6 w-6" />
      <span class="truncate text-[0.9375rem] font-semibold tracking-tight text-ink">
        laya<span class="text-accent">-rs</span>
      </span>
    </span>
  );
}

export function ThemeToggle(props: { class?: string }) {
  return (
    <button
      type="button"
      onClick={toggleTheme}
      title={theme() === "dark" ? "Switch to light" : "Switch to dark"}
      aria-label={theme() === "dark" ? "Switch to light theme" : "Switch to dark theme"}
      class={`inline-flex h-8 w-8 items-center justify-center rounded-lg text-muted transition-colors hover:bg-surface-2 hover:text-ink ${
        props.class ?? ""
      }`}
    >
      <Show
        when={theme() === "dark"}
        fallback={
          <svg viewBox="0 0 24 24" class="h-4 w-4" fill="currentColor" aria-hidden="true">
            <path d="M12 3a9 9 0 1 0 9 9c0-.34-.02-.67-.05-1A7 7 0 0 1 13 4.05c-.33-.03-.66-.05-1-.05Z" />
          </svg>
        }
      >
        <svg viewBox="0 0 24 24" class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">
          <circle cx="12" cy="12" r="4" />
          <path
            stroke-linecap="round"
            d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.1 5.1l1.4 1.4M17.5 17.5l1.4 1.4M18.9 5.1l-1.4 1.4M6.5 17.5l-1.4 1.4"
          />
        </svg>
      </Show>
    </button>
  );
}
