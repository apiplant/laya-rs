import { For, type ParentProps } from "solid-js";
import { useLocation } from "@solidjs/router";

interface NavItem {
  label: string;
  href: string;
}

interface NavGroup {
  title: string;
  items: NavItem[];
}

const NAV: NavGroup[] = [
  {
    title: "Get started",
    items: [
      { label: "Overview", href: "/docs" },
      { label: "As a library", href: "/docs/library" },
      { label: "laya CLI", href: "/docs/cli" },
    ],
  },
  {
    title: "Advanced",
    items: [{ label: "Training (RLCD)", href: "/docs/training" }],
  },
];

function SidebarLink(props: { item: NavItem; active: boolean }) {
  return (
    <a
      href={props.item.href}
      class={`block rounded-md px-3 py-1.5 text-sm transition-colors ${
        props.active
          ? "bg-accent-soft font-medium text-accent"
          : "text-muted hover:bg-surface-2 hover:text-ink"
      }`}
    >
      {props.item.label}
    </a>
  );
}

function Sidebar() {
  const location = useLocation();
  const active = (href: string) =>
    href === "/docs" ? location.pathname === "/docs" : location.pathname.startsWith(href);

  return (
    <nav class="space-y-6">
      <For each={NAV}>
        {(group) => (
          <div>
            <h2 class="px-3 text-[0.6875rem] font-semibold uppercase tracking-[0.12em] text-faint">
              {group.title}
            </h2>
            <div class="mt-1.5 space-y-0.5">
              <For each={group.items}>{(item) => <SidebarLink item={item} active={active(item.href)} />}</For>
            </div>
          </div>
        )}
      </For>
    </nav>
  );
}

export function DocsLayout(props: ParentProps) {
  return (
    <div class="mx-auto w-full max-w-6xl px-5 py-10 sm:py-12">
      <div class="grid gap-8 lg:grid-cols-[14rem_minmax(0,1fr)] lg:gap-12">
        <aside class="lg:sticky lg:top-20 lg:h-fit">
          <Sidebar />
        </aside>
        <div class="min-w-0 pb-16">{props.children}</div>
      </div>
    </div>
  );
}
