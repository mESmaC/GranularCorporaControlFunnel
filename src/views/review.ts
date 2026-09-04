import { api, type SampleView, type SessionInfo, type StatusCounts, type ReviewMode } from "../api";
import { h, clear, fmtBytes, fmtInt, toast, guard, on, statusBar } from "../ui";
import { state, setSession, refreshTags, navigate, type View } from "../state";

let root: HTMLElement;
let sessions: SessionInfo[] = [];
let sessionId: number | null = null;
let mode: ReviewMode = "pending";
let current: SampleView | null = null;
let history: number[] = [];
let shownAt = 0;
let counts: StatusCounts | null = null;
let pretty = true;
let wrap = true;
let busy = false;
let unsubs: (() => void)[] = [];

// Elements re-rendered independently.
let topEl: HTMLElement;
let bodyEl: HTMLElement;
let actionsEl: HTMLElement;

function session(): SessionInfo | undefined {
  return sessions.find((s) => s.id === sessionId);
}

async function refreshCounts() {
  if (!sessionId) return;
  counts = (await guard(api.sessionCounts(sessionId))) ?? counts;
}

async function loadNext(afterOrdinal: number | null) {
  if (!sessionId) {
    current = null;
    return;
  }
  current = (await guard(api.nextSample(sessionId, mode, afterOrdinal), "next sample")) ?? null;
  shownAt = Date.now();
}

async function show(id: number) {
  current = (await guard(api.getSample(id), "load sample")) ?? null;
  shownAt = Date.now();
}

async function decide(verdict: "good" | "bad" | "skip") {
  if (!current || busy) return;
  busy = true;
  const s = current;
  const viewMs = Date.now() - shownAt;
  const card = bodyEl.querySelector(".sample-card");
  card?.classList.add(verdict === "bad" ? "swipe-left" : verdict === "good" ? "swipe-right" : "swipe-down");
  const res = await guard(api.decideSample(s.id, verdict, viewMs), "save decision");
  if (res) counts = res;
  if (history[history.length - 1] !== s.id) history.push(s.id);
  if (history.length > 500) history.shift();
  await new Promise((r) => setTimeout(r, 150));
  await loadNext(s.ordinal);
  busy = false;
  renderAll();
}

async function prev() {
  if (busy) return;
  let id = history.pop();
  if (id !== undefined && current && id === current.id) id = history.pop();
  if (id === undefined) return toast("Nothing to go back to", "info", 1500);
  await show(id);
  renderAll();
}

async function next() {
  if (busy || !current) return;
  if (history[history.length - 1] !== current.id) history.push(current.id);
  await loadNext(current.ordinal);
  renderAll();
}

async function toggleTag(tagId: number) {
  if (!current) return;
  const on = current.tags.includes(tagId);
  const res = await guard(api.setSampleTag(current.id, tagId, !on), "tag");
  if (res) current.tags = res;
  renderTags();
}

async function addTag(name: string) {
  if (!current) return;
  const n = name.trim();
  if (!n) return;
  const id = await guard(api.createTag(n), "create tag");
  if (id === undefined) return;
  await refreshTags();
  const res = await guard(api.setSampleTag(current.id, id, true), "tag");
  if (res) current.tags = res;
  renderTags();
}

function renderTop() {
  clear(topEl);
  const sel = h(
    "select",
    { onChange: async () => { sessionId = Number(sel.value) || null; setSession(sessionId); history = []; await refreshCounts(); await loadNext(null); renderAll(); } },
    h("option", { value: "" }, "— session —"),
    ...sessions.filter((s) => s.status === "ready").map((s) => h("option", { value: String(s.id), selected: s.id === sessionId }, `${s.name}  (${s.counts.pending} pending)`)),
  );
  const modeSel = h(
    "select",
    { onChange: async () => { mode = modeSel.value as ReviewMode; history = []; await loadNext(null); renderAll(); } },
    h("option", { value: "pending", selected: mode === "pending" }, "Pending"),
    h("option", { value: "skip", selected: mode === "skip" }, "Skipped queue"),
    h("option", { value: "decided", selected: mode === "decided" }, "Decided (re-review)"),
    h("option", { value: "all", selected: mode === "all" }, "Everything"),
  );
  const c = counts;
  topEl.appendChild(h("div", { style: { display: "contents" } },
    sel,
    modeSel,
    c
      ? h("div", { class: "review-counts" },
          statusBar(c),
          h("span", null, h("b", { style: { color: "var(--good)" } }, fmtInt(c.good)), " good"),
          h("span", null, h("b", { style: { color: "var(--bad)" } }, fmtInt(c.bad)), " bad"),
          h("span", null, h("b", { style: { color: "var(--skip)" } }, fmtInt(c.skip)), " skipped"),
          h("span", null, h("b", null, fmtInt(c.pending)), " pending"),
        )
      : null,
    h("span", { style: { flex: "1" } }),
    h("button", { class: `sm ${pretty ? "" : "ghost"}`, title: "Render JSON rows as fields (p)", onClick: () => { pretty = !pretty; renderBody(); } }, "Fields"),
    h("button", { class: `sm ${wrap ? "" : "ghost"}`, title: "Soft-wrap long lines (w)", onClick: () => { wrap = !wrap; renderBody(); } }, "Wrap"),
  ));
}

function locatorText(s: SampleView): string {
  const l = s.locator ?? {};
  switch (l.type) {
    case "row":
      return `row ${fmtInt(l.index)}`;
    case "bytes":
      return `bytes ${fmtInt(l.byte_start)}–${fmtInt(l.byte_end)} · ${l.n_lines} line${l.n_lines === 1 ? "" : "s"}`;
    case "stream_lines":
      return `line ${fmtInt(l.line_start)} +${l.n_lines} (decompressed)`;
    default:
      return s.locator_kind;
  }
}

function parseRow(text: string): Record<string, unknown> | null {
  const t = text.trim();
  if (!t.startsWith("{")) return null;
  try {
    const v = JSON.parse(t);
    return v && typeof v === "object" && !Array.isArray(v) ? v : null;
  } catch {
    return null;
  }
}

function renderContent(s: SampleView): HTMLElement {
  if (pretty && s.kind === "rows") {
    const obj = parseRow(s.text);
    if (obj) {
      const fields = Object.entries(obj).map(([k, v]) =>
        h("div", { class: "fld" }, h("div", { class: "fld-key" }, k), h("pre", { class: "fld-val" }, typeof v === "string" ? v : JSON.stringify(v, null, 2))),
      );
      return h("div", { class: "sample-fields" }, ...fields);
    }
  }
  return h("pre", { class: `sample-text ${wrap ? "" : "nowrap"}` }, s.text);
}

function renderBody() {
  clear(bodyEl);
  const s = current;
  if (!s) {
    const c = counts;
    bodyEl.appendChild(
      h("div", { class: "empty", style: { gridColumn: "1 / -1", alignSelf: "start" } },
        sessionId ? h("div", null, h("div", { style: { fontSize: "16px", marginBottom: "8px" } }, mode === "pending" ? "No pending samples left in this session." : mode === "skip" ? "The skip queue is empty." : "Nothing here."),
          h("div", { class: "row", style: { justifyContent: "center" } },
            c && c.skip > 0 && mode !== "skip" ? h("button", { onClick: async () => { mode = "skip"; await loadNext(null); renderAll(); } }, `Review ${c.skip} skipped`) : null,
            h("button", { onClick: () => navigate("sessions") }, "Sessions"),
            h("button", { onClick: () => navigate("receipt") }, "Write receipt"),
          ))
        : h("div", null, "Pick a session above, or build one under Sessions."),
      ),
    );
    return;
  }
  const status = s.status !== "pending" ? h("span", { class: `status-banner ${s.status}` }, `already marked ${s.status}${s.decided_by ? ` by ${s.decided_by}` : ""} — deciding again overrides`) : null;
  const pred = s.prediction
    ? h("div", { class: "prediction" },
        h("span", { class: "who" }, `model ${s.prediction.model_id}:`),
        h("span", { class: `pill ${s.prediction.verdict === "good" ? "good" : s.prediction.verdict === "bad" ? "bad" : ""}` }, `${s.prediction.verdict} ${(100 * s.prediction.score).toFixed(0)}%`),
        ...s.prediction.tags.map((t) => h("span", { class: "pill" }, t)),
        s.prediction.tags.length
          ? h("button", { class: "ghost sm", onClick: async () => {
              for (const name of s.prediction!.tags) {
                const tag = state.tags.find((t) => t.name === name);
                const id = tag ? tag.id : await guard(api.createTag(name));
                if (id !== undefined && !s.tags.includes(id)) s.tags = (await guard(api.setSampleTag(s.id, id, true))) ?? s.tags;
              }
              await refreshTags();
              renderTags();
            } }, "use tags")
          : null,
      )
    : null;
  const card = h("div", { class: "sample-card enter" },
    h("div", { class: "sample-head" },
      h("div", { class: "sample-path", title: "Reveal in Explorer", onClick: () => guard(api.revealFile(s.file_id), "reveal") }, s.abs_path),
      h("div", { class: "sample-meta" },
        h("span", { class: "pill accent" }, s.format),
        h("span", { class: "pill" }, s.kind),
        s.compression !== "none" ? h("span", { class: "pill" }, s.compression) : null,
        h("span", { class: "pill" }, locatorText(s)),
        h("span", { class: "pill" }, `file ${fmtBytes(s.file_size)}`),
        h("span", { class: "pill" }, `sample ${fmtBytes(s.text_bytes)}${s.truncated ? " (truncated)" : ""}`),
        s.duplicates > 0 ? h("span", { class: "pill bad", title: "Identical text exists in other samples" }, `dup ×${s.duplicates}`) : null,
        h("span", { class: "pill" }, `#${s.ordinal + 1}`),
        status,
      ),
      pred,
    ),
    renderContent(s),
  );
  bodyEl.append(card, h("div", { class: "tag-panel" }));
  renderTags();
}

function renderTags() {
  const panel = bodyEl.querySelector(".tag-panel");
  if (!panel || !current) return;
  clear(panel);
  const s = current;
  const list = h("div", { class: "tag-list" });
  const sorted = [...state.tags].sort((a, b) => (a.hotkey ?? 99) - (b.hotkey ?? 99) || a.name.localeCompare(b.name));
  for (const t of sorted) {
    const on = s.tags.includes(t.id);
    list.appendChild(
      h("div", { class: `tag-item ${on ? "on" : ""}`, onClick: () => toggleTag(t.id) },
        h("span", { class: "hk" }, t.hotkey ? String(t.hotkey) : ""),
        h("span", { class: "name" }, t.name),
        h("span", { class: "src" }, on ? "✓" : ""),
      ),
    );
  }
  const input = h("input", { type: "text", placeholder: "new tag… (Enter)" });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      addTag(input.value);
      input.value = "";
      input.blur();
    } else if (e.key === "Escape") {
      input.blur();
    }
    e.stopPropagation();
  });
  panel.append(
    h("div", { class: "row between" }, h("b", null, "Tags"), h("span", { class: "small muted" }, "1–9 toggle")),
    list,
    h("div", { class: "tag-add" }, input, h("button", { class: "sm", onClick: () => { addTag(input.value); input.value = ""; } }, "＋")),
    h("div", { class: "small muted" }, "Tags are saved instantly and snapshotted with the verdict."),
  );
}

function renderActions() {
  clear(actionsEl);
  const has = !!current;
  actionsEl.append(
    h("button", { disabled: history.length === 0, onClick: prev, title: "Backspace / ←" }, "◀ Prev"),
    h("span", { class: "spacer" }),
    h("button", { class: "btn-bad", disabled: !has, onClick: () => decide("bad") }, h("kbd", null, "−"), " Bad"),
    h("button", { class: "btn-skip", disabled: !has, onClick: () => decide("skip"), title: "No hotkey on purpose: park it in the queue for later" }, "Skip"),
    h("button", { class: "btn-good", disabled: !has, onClick: () => decide("good") }, h("kbd", null, "="), " Good"),
    h("span", { class: "spacer" }),
    h("button", { disabled: !has, onClick: () => guard(api.revealFile(current!.file_id), "reveal"), title: "Open Explorer at the file so you can edit it safely outside GCCF" }, "Reveal file"),
    h("button", { disabled: !has, onClick: () => guard(api.openFile(current!.file_id), "open"), title: "Open with the default application" }, "Open"),
    h("button", { disabled: !has, onClick: next, title: "→ (peek without deciding)" }, "Next ▶"),
  );
}

function renderAll() {
  renderTop();
  renderBody();
  renderActions();
}

export const reviewView: View = {
  async render(el, params) {
    root = el;
    unsubs.forEach((u) => u());
    unsubs = [];
    sessions = (await guard(api.listSessions(), "sessions")) ?? [];
    if (params?.sessionId) setSession(params.sessionId);
    if (params?.mode) mode = params.mode;
    sessionId = state.sessionId;
    if (sessionId && !sessions.some((s) => s.id === sessionId && s.status === "ready")) sessionId = null;
    if (!sessionId) {
      const first = sessions.find((s) => s.status === "ready" && s.counts.pending > 0) ?? sessions.find((s) => s.status === "ready");
      sessionId = first?.id ?? null;
      setSession(sessionId);
    }
    history = [];
    current = null;
    await refreshTags();
    await refreshCounts();
    if (params?.sampleId) await show(params.sampleId);
    else await loadNext(null);

    topEl = h("div", { class: "review-top" });
    bodyEl = h("div", { class: "review-body" });
    actionsEl = h("div", { class: "review-actions" });
    root.appendChild(h("div", { class: "review" }, topEl, h("div", { class: "small muted", style: { marginTop: "-6px" } }, session() ? `${session()!.name} · ${session()!.root_path}` : ""), bodyEl, actionsEl));
    renderAll();

    unsubs.push(on("tags", () => renderTags()));
    unsubs.push(on("job-finished", async (j) => {
      if (j.kind === "session") {
        sessions = (await guard(api.listSessions())) ?? sessions;
        renderTop();
      }
    }));
  },
  onKey(e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    switch (e.key) {
      case "-":
      case "_":
        e.preventDefault();
        decide("bad");
        break;
      case "=":
      case "+":
        e.preventDefault();
        decide("good");
        break;
      case "Backspace":
      case "ArrowLeft":
        e.preventDefault();
        prev();
        break;
      case "ArrowRight":
        e.preventDefault();
        next();
        break;
      case "p":
        pretty = !pretty;
        renderBody();
        break;
      case "w":
        wrap = !wrap;
        renderBody();
        break;
      default:
        if (e.key >= "1" && e.key <= "9") {
          const t = state.tags.find((t) => t.hotkey === Number(e.key));
          if (t) {
            e.preventDefault();
            toggleTag(t.id);
          }
        }
    }
  },
  destroy() {
    unsubs.forEach((u) => u());
    unsubs = [];
  },
};
