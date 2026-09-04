// Small DOM + formatting helpers shared by every view. No framework: the UI is
// a handful of screens and a keyboard-driven review card.

type Child = Node | string | number | null | undefined | false | Child[];

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, any> | null = null,
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") el.className = String(v);
      else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
      else if (k === "dataset") Object.assign(el.dataset, v);
      else if (k in el && k !== "list") (el as any)[k] = v;
      else el.setAttribute(k, String(v));
    }
  }
  append(el, children);
  return el;
}

function append(el: Node, children: Child[]) {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    if (Array.isArray(c)) append(el, c);
    else if (c instanceof Node) el.appendChild(c);
    else el.appendChild(document.createTextNode(String(c)));
  }
}

export function clear(el: Element) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

export function fmtBytes(n: number): string {
  if (!isFinite(n) || n < 0) return "–";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v < 10 && i > 0 ? v.toFixed(2) : v < 100 && i > 0 ? v.toFixed(1) : Math.round(v)} ${units[i]}`;
}

export function fmtInt(n: number): string {
  return n.toLocaleString();
}

export function fmtPct(a: number, b: number): string {
  if (!b) return "–";
  return `${((100 * a) / b).toFixed(1)}%`;
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "–";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function fmtDuration(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s - m * 60)}s`;
}

// --- toasts ----------------------------------------------------------------

let toastHost: HTMLElement | null = null;

export function toast(message: string, kind: "info" | "ok" | "err" = "info", ms = 3200) {
  if (!toastHost) toastHost = document.getElementById("toasts");
  if (!toastHost) return;
  const el = h("div", { class: `toast toast-${kind}` }, message);
  toastHost.appendChild(el);
  requestAnimationFrame(() => el.classList.add("show"));
  setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => el.remove(), 250);
  }, ms);
}

export function errMsg(e: unknown): string {
  if (typeof e === "string") return e;
  if (e && typeof e === "object" && "message" in e) return String((e as any).message);
  return String(e);
}

export async function guard<T>(p: Promise<T>, what?: string): Promise<T | undefined> {
  try {
    return await p;
  } catch (e) {
    toast(`${what ? what + ": " : ""}${errMsg(e)}`, "err", 6000);
    console.error(e);
    return undefined;
  }
}

// --- tiny event bus --------------------------------------------------------

type Handler = (payload: any) => void;
const handlers = new Map<string, Set<Handler>>();

export function on(event: string, fn: Handler): () => void {
  if (!handlers.has(event)) handlers.set(event, new Set());
  handlers.get(event)!.add(fn);
  return () => handlers.get(event)?.delete(fn);
}

export function emit(event: string, payload?: any) {
  handlers.get(event)?.forEach((fn) => {
    try {
      fn(payload);
    } catch (e) {
      console.error(e);
    }
  });
}

// --- form helpers ----------------------------------------------------------

export function numberInput(opts: { value: number; min?: number; max?: number; step?: number; width?: string; onChange?: (v: number) => void; id?: string }) {
  const el = h("input", {
    type: "number",
    value: String(opts.value),
    min: opts.min,
    max: opts.max,
    step: opts.step ?? 1,
    id: opts.id,
    style: opts.width ? { width: opts.width } : undefined,
  });
  if (opts.onChange) el.addEventListener("change", () => opts.onChange!(Number(el.value)));
  return el;
}

export function field(label: string, control: Node, hint?: string) {
  return h("label", { class: "field" }, h("span", { class: "field-label" }, label), control, hint ? h("span", { class: "hint" }, hint) : null);
}

export function progressBar(done: number, total: number, cls = "") {
  const pct = total > 0 ? Math.min(100, (100 * done) / total) : 0;
  return h(
    "div",
    { class: `bar ${cls}` },
    h("div", { class: "bar-fill", style: { width: `${pct}%` } }),
  );
}

export function statusBar(counts: { good: number; bad: number; skip: number; pending: number }) {
  const total = counts.good + counts.bad + counts.skip + counts.pending;
  const seg = (n: number, cls: string) => h("div", { class: `seg ${cls}`, style: { width: total ? `${(100 * n) / total}%` : "0%" }, title: `${cls}: ${n}` });
  return h("div", { class: "bar stacked" }, seg(counts.good, "good"), seg(counts.bad, "bad"), seg(counts.skip, "skip"), seg(counts.pending, "pending"));
}

export function confirmDialog(message: string): boolean {
  return window.confirm(message);
}

export function debounce<F extends (...a: any[]) => void>(fn: F, ms: number): F {
  let t: number | undefined;
  return ((...args: any[]) => {
    if (t) window.clearTimeout(t);
    t = window.setTimeout(() => fn(...args), ms);
  }) as F;
}

export function isTypingTarget(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  const tag = t.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t.isContentEditable;
}
