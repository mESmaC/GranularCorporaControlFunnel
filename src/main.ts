import { listen } from "@tauri-apps/api/event";
import { api, type JobProgress } from "./api";
import { h, clear, on, emit, progressBar, toast, errMsg, isTypingTarget } from "./ui";
import { state, refreshTags, refreshSettings, type View } from "./state";
import { rootsView } from "./views/roots";
import { sessionsView } from "./views/sessions";
import { reviewView } from "./views/review";
import { queueView } from "./views/queue";
import { tagsView } from "./views/tags";
import { classifierView } from "./views/classifier";
import { receiptView } from "./views/receipt";
import { settingsView } from "./views/settings";

interface NavItem {
  name: string;
  label: string;
  icon: string;
  view: View;
  key?: string;
}

const NAV: NavItem[] = [
  { name: "roots", label: "Corpora", icon: "▤", view: rootsView, key: "F1" },
  { name: "sessions", label: "Sessions", icon: "◫", view: sessionsView, key: "F2" },
  { name: "review", label: "Review", icon: "⇄", view: reviewView, key: "F3" },
  { name: "queue", label: "Queue", icon: "☰", view: queueView, key: "F4" },
  { name: "tags", label: "Tags", icon: "#", view: tagsView, key: "F5" },
  { name: "classifier", label: "Classifier", icon: "⚙", view: classifierView, key: "F6" },
  { name: "receipt", label: "Receipt", icon: "⬇", view: receiptView, key: "F7" },
  { name: "settings", label: "Settings", icon: "…", view: settingsView },
];

let active: NavItem | null = null;
const jobs = new Map<number, JobProgress>();

const navEl = document.getElementById("nav")!;
const viewEl = document.getElementById("view")!;
const jobsEl = document.getElementById("jobs")!;
const footerEl = document.getElementById("footer")!;

function renderNav() {
  clear(navEl);
  for (const item of NAV) {
    const btn = h(
      "button",
      { class: active?.name === item.name ? "active" : "", onClick: () => go(item.name) },
      h("span", { class: "icon" }, item.icon),
      item.label,
      item.key ? h("span", { class: "key" }, item.key) : null,
    );
    navEl.appendChild(btn);
  }
}

async function go(name: string, params?: any) {
  const item = NAV.find((n) => n.name === name);
  if (!item) return;
  if (active?.view.destroy) {
    try {
      active.view.destroy();
    } catch (e) {
      console.error(e);
    }
  }
  active = item;
  renderNav();
  clear(viewEl);
  viewEl.scrollTop = 0;
  try {
    await item.view.render(viewEl, params);
  } catch (e) {
    console.error(e);
    viewEl.appendChild(h("div", { class: "empty" }, `Failed to render ${item.label}: ${errMsg(e)}`));
  }
}

function renderJobs() {
  clear(jobsEl);
  const list = [...jobs.values()].sort((a, b) => b.id - a.id).slice(0, 8);
  for (const j of list) {
    const running = j.status === "running";
    const el = h(
      "div",
      { class: `job ${j.status}` },
      h(
        "div",
        { class: "job-title" },
        h("span", { class: "job-kind" }, j.kind === "scan" ? "scan" : j.kind === "session" ? "sampling" : j.kind),
        running
          ? h("button", { class: "ghost sm", title: "Cancel", onClick: () => api.cancelJob(j.id) }, "✕")
          : h("button", { class: "ghost sm", title: "Dismiss", onClick: () => { jobs.delete(j.id); renderJobs(); } }, "–"),
      ),
      running && j.total > 0 ? progressBar(j.done, j.total) : null,
      h("div", { class: "job-msg", title: j.message }, j.message || (running ? "working…" : j.status)),
    );
    jobsEl.appendChild(el);
  }
}

async function init() {
  renderNav();
  try {
    state.info = await api.appInfo();
    footerEl.textContent = `v${state.info.version} · ${state.info.cpus} cores · ${state.info.db_path}`;
  } catch (e) {
    footerEl.textContent = `backend unavailable: ${errMsg(e)}`;
  }
  await Promise.all([refreshTags().catch(() => []), refreshSettings().catch(() => ({}))]);

  try {
    for (const j of await api.listJobs()) jobs.set(j.id, j);
    renderJobs();
  } catch {
    /* ignore */
  }

  await listen<JobProgress>("job:progress", (ev) => {
    const j = ev.payload;
    const prev = jobs.get(j.id);
    jobs.set(j.id, j);
    renderJobs();
    emit("job", j);
    if (prev?.status === "running" && j.status !== "running") {
      toast(`${j.kind}: ${j.message}`, j.status === "done" ? "ok" : j.status === "error" ? "err" : "info");
      emit("job-finished", j);
    }
  });

  on("navigate", ({ view, params }) => go(view, params));

  window.addEventListener("keydown", (e) => {
    if (e.key.startsWith("F") && e.key.length <= 3) {
      const item = NAV.find((n) => n.key === e.key);
      if (item) {
        e.preventDefault();
        go(item.name);
        return;
      }
    }
    if (isTypingTarget(e.target)) return;
    active?.view.onKey?.(e);
  });

  const start = state.sessionId ? "review" : "roots";
  await go(start);
}

init().catch((e) => {
  console.error(e);
  viewEl.appendChild(h("div", { class: "empty" }, `Startup failed: ${errMsg(e)}`));
});
