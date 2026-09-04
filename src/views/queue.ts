import { api, type SessionInfo, type SampleSummary } from "../api";
import { h, clear, fmtDate, guard, on } from "../ui";
import { state, refreshTags, navigate, setSession, type View } from "../state";

let listEl: HTMLElement;
let sessions: SessionInfo[] = [];
let sessionFilter: number | null = null;
let status: "skip" | "good" | "bad" = "skip";
let unsub: (() => void) | null = null;

function tagNames(ids: number[]): string {
  return ids.map((id) => state.tags.find((t) => t.id === id)?.name ?? `#${id}`).join(", ");
}

async function refresh() {
  const rows = (await guard(api.listSamples(sessionFilter, [status], 500, 0), "queue")) ?? [];
  clear(listEl);
  if (!rows.length) {
    listEl.appendChild(h("div", { class: "empty" }, status === "skip" ? "The skip queue is empty. Samples you skip while reviewing land here for a second look." : `No samples marked ${status}.`));
    return;
  }
  const row = (s: SampleSummary) =>
    h(
      "tr",
      null,
      h("td", { class: "small muted", style: { whiteSpace: "nowrap" } }, s.session_name),
      h("td", null, h("div", { class: "path" }, s.rel_path), h("div", { class: "small muted" }, `${s.format} · #${s.ordinal + 1}${s.tags.length ? " · " + tagNames(s.tags) : ""}`)),
      h("td", null, h("div", { class: "mono small", style: { maxHeight: "60px", overflow: "hidden", whiteSpace: "pre-wrap", wordBreak: "break-word", color: "var(--text-2)" } }, s.preview)),
      h("td", { class: "small muted", style: { whiteSpace: "nowrap" } }, fmtDate(s.decided_at)),
      h(
        "td",
        null,
        h("button", { class: "sm primary", onClick: () => { setSession(s.session_id); navigate("review", { sessionId: s.session_id, sampleId: s.id, mode: status === "skip" ? "skip" : "decided" }); } }, "Review"),
      ),
    );
  listEl.appendChild(
    h("table", null,
      h("thead", null, h("tr", null, h("th", null, "session"), h("th", null, "file"), h("th", null, "preview"), h("th", null, "when"), h("th", null, ""))),
      h("tbody", null, ...rows.map(row)),
    ),
  );
}

export const queueView: View = {
  async render(el) {
    sessions = (await guard(api.listSessions(), "sessions")) ?? [];
    await refreshTags();
    if (sessionFilter && !sessions.some((s) => s.id === sessionFilter)) sessionFilter = null;
    const sel = h(
      "select",
      { onChange: () => { sessionFilter = Number(sel.value) || null; refresh(); } },
      h("option", { value: "" }, "all sessions"),
      ...sessions.map((s) => h("option", { value: String(s.id), selected: s.id === sessionFilter }, `${s.name} (${s.counts.skip} skipped)`)),
    );
    const chip = (v: typeof status, label: string) =>
      h("span", { class: `chip ${status === v ? "on" : ""}`, onClick: () => { status = v; render(); } }, label);
    const chips = h("div", { class: "chips" }, chip("skip", "Skipped"), chip("bad", "Marked bad"), chip("good", "Marked good"));
    const render = () => {
      chips.querySelectorAll(".chip").forEach((c, i) => c.classList.toggle("on", ["skip", "bad", "good"][i] === status));
      refresh();
    };
    listEl = h("div");
    el.append(
      h("h1", null, "Queue"),
      h("p", { class: "lead" }, "Skipped samples wait here. Pick one to reopen it in the reviewer, decide, and it leaves the queue. You can also revisit anything already marked good or bad."),
      h("div", { class: "row", style: { marginBottom: "14px" } }, sel, chips),
      listEl,
    );
    await refresh();
    unsub = on("job-finished", () => refresh());
  },
  destroy() {
    unsub?.();
    unsub = null;
  },
};
