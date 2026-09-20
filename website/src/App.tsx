import { type ParentProps } from "solid-js";
import { createRouter } from "@solidjs/router";
import { Header } from "./components/Header";
import { Footer } from "./components/Footer";
import { Home } from "./components/Home";
import { DocsOverview } from "./components/docs/Overview";
import { DocsLibrary } from "./components/docs/Library";
import { DocsCli } from "./components/docs/Cli";
import { DocsTraining } from "./components/docs/Training";
import { Demo } from "./components/demo/Demo";
import { LinkButton } from "./components/ui";

function Shell(props: ParentProps) {
  return (
    <div class="flex min-h-screen flex-col">
      <Header />
      <main class="flex-1">{props.children}</main>
      <Footer />
    </div>
  );
}

function NotFound() {
  return (
    <div class="mx-auto w-full max-w-6xl px-5 py-28 text-center">
      <p class="font-mono text-sm text-accent">404</p>
      <h1 class="mt-3 text-3xl font-semibold tracking-tight text-ink">No such page</h1>
      <p class="mx-auto mt-3 max-w-md leading-relaxed text-muted">
        The link may be old, or the page may have been renamed.
      </p>
      <div class="mt-8 flex justify-center gap-3">
        <LinkButton href="/" variant="primary">
          Home
        </LinkButton>
      </div>
    </div>
  );
}

const Router = createRouter({
  routes: [
    { path: "/", component: Home },
    { path: "/docs", component: DocsOverview },
    { path: "/docs/library", component: DocsLibrary },
    { path: "/docs/cli", component: DocsCli },
    { path: "/docs/training", component: DocsTraining },
    { path: "/demo", component: Demo },
    { path: "*404", component: NotFound },
  ],
});

export const { paths } = Router;

export function App() {
  return <Router>{(props) => <Shell>{props.children}</Shell>}</Router>;
}
