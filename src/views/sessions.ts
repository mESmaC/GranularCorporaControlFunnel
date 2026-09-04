import { api, type RootInfo, type FormatStat, type SessionInfo } from "../api";
import { h, clear, fmtBytes, fmtInt, fmtDate, toast, guard, on, field, statusBar, confirmDialog } from "../ui";
import { state, setSession, navigate, type View } from "../state";

let listEl: HTMLElement;
let formEl: HTMLElement;
let unsub: (() => void) | null = null;
let roots: RootInfo[] = [];
let selectedRoot: number | null = null;
let formatStats: FormatStat[] = [];
const selectedFormats = new Set<string>();

function eligibleCount(): { files: number; bytes: number } {
  let files = 0;
  let bytes = 0;
  for (const s of formatStats) {
    if (s.kind === "unsupported") continue;
    if (selectedFormats.size && !selectedFormats.has(s.format)) continue;
    files += s.count;
    bytes += s.bytes;
  }
  return { files, bytes };
}

async function renderForm() {
  clear(formEl);
  const rootSel = h(
    "select",
    { onChange: async () => { selectedRoot = Number(rootSel.value) || null; selectedFormats.clear(); await loadFormats(); renderForm(); } },
    h("option", { value: "" }, "— choose a corpus —"),
    ...roots.map((r) => h("option", { value: String(r.id), selected: r.id === selectedRoot }, `${r.path}  (${fmtInt(r.sampleable_files)} sampleable files)`)),
  );
  const filesIn = h("input", { type: "number", min: 1, value: "20" });
  const perFileIn = h("input", { type: "number", min: 1, value: "50" });
  const linesIn = h("input", { type: "number", min: 1, value: "50" });
  const seedIn = h("input", { type: "number", placeholder: "random", style: { width: "130px" } });
  const nameIn = h("input", { type: "text", placeholder: "optional name", style: { width: "260px" } });
  const excludeIn = h("input", { type: "checkbox", checked: true });
  const maxBytesIn = h("input", { type: "number", min: 1, value: String(Math.round(Number(state.settings["max_bytes_per_sample"] ?? 262144) / 1024)) });
  const streamCapIn = h("input", { type: "number", min: 0, step: 0.5, value: String(Number(state.settings["max_stream_bytes"] ?? 0) / 1e9) });
  const threadsIn = h("input", { type: "number", min: 1, max: 64, value: String(state.info?.default_threads ?? 4) });

  const eligibleEl = h("div", { class: "small muted" });
  const updateEligible = () => {
    const e = eligibleCount();
    const files = Number(filesIn.value) || 0;
    const per = Number(perFileIn.value) || 0;
    eligibleEl.textContent = selectedRoot
      ? `${fmtInt(e.files)} eligible files (${fmtBytes(e.bytes)}) · this session will open ${fmtInt(Math.min(files, e.files))} of them and cut up to ${fmtInt(Math.min(files, e.files) * per)} samples`
      : "";
  };
  filesIn.addEventListener("input", updateEligible);
  perFileIn.addEventListener("input", updateEligible);

  const chips = h("div", { class: "chips" });
  const sampleable = formatStats.filter((s) => s.kind !== "unsupported");
  const byFormat = new Map<string, { count: number; bytes: number }>();
  for (const s of sampleable) {
    const cur = byFormat.get(s.format) ?? { count: 0, bytes: 0 };
    cur.count += s.count;
    cur.bytes += s.bytes;
    byFormat.set(s.format, cur);
  }
  for (const [fmt, c] of [...byFormat.entries()].sort((a, b) => b[1].count - a[1].count)) {
    const chip = h(
      "span",
      { class: `chip ${selectedFormats.has(fmt) ? "on" : ""}`, onClick: () => { if (selectedFormats.has(fmt)) selectedFormats.delete(fmt); else selectedFormats.add(fmt); chip.classList.toggle("on"); updateEligible(); } },
      fmt,
      h("span", { class: "cnt" }, `${fmtInt(c.count)} · ${fmtBytes(c.bytes)}`),
    );
    chips.appendChild(chip);
  }
  updateEligible();

  const submit = async () => {
    if (!selectedRoot) return toast("Choose a corpus first", "err");
    const seed = seedIn.value.trim() === "" ? undefined : Number(seedIn.value);
    const res = await guard(
      api.createSession({
        root_id: selectedRoot,
        name: nameIn.value.trim() || undefined,
        files: Number(filesIn.value),
        samples_per_file: Number(perFileIn.value),
        lines_per_sample: Number(linesIn.value),
        seed,
        formats: [...selectedFormats],
        exclude_reviewed_files: excludeIn.checked,
        max_bytes_per_sample: Math.max(1, Number(maxBytesIn.value)) * 1024,
        max_stream_bytes: Math.round(Number(streamCapIn.value) * 1e9),
        threads: Number(threadsIn.value) || undefined,
      }),
      "create session",
    );
    if (res?.session_id) {
      setSession(res.session_id);
      toast("Building session — it will appear below as samples are cut.", "ok");
      await refreshList();
    }
  };

  formEl.append(
    h("div", { class: "card-title" }, "New review session"),
    h("div", { class: "stack" },
      field("Corpus", rootSel),
      selectedRoot ? field("Formats (none selected = all sampleable formats)", chips) : null,
      h("div", { class: "form-grid" },
        field("Files to open", filesIn, "random selection from eligible files"),
        field("Samples per file", perFileIn, "rows for parquet/jsonl/csv"),
        field("Lines per sample", linesIn, "for plain text / code / markdown"),
        field("Seed", seedIn, "same seed + same corpus = same picks"),
        field("Name", nameIn),
      ),
      h("label", { class: "check" }, excludeIn, "Skip files that already have samples in another session"),
      h("details", null,
        h("summary", null, "Advanced"),
        h("div", { class: "form-grid", style: { marginTop: "8px" } },
          field("Max KB kept per sample", maxBytesIn, "longer samples are cut and flagged"),
          field("Compressed stream cap (GB)", streamCapIn, "0 = read the whole .gz/.zst file"),
          field("Sampling threads", threadsIn, `up to ${state.info?.cpus ?? "?"} cores`),
        ),
      ),
      eligibleEl,
      h("div", { class: "row" }, h("button", { class: "primary", onClick: submit }, "Build session")),
    ),
  );
}

async function loadFormats() {
  formatStats = selectedRoot ? (await guard(api.rootFormatStats(selectedRoot), "formats")) ?? [] : [];
}

function sessionRow(s: SessionInfo): HTMLElement {
  const decided = s.counts.good + s.counts.bad + s.counts.skip;
  const nameIn = h("input", { class: "inline-edit", value: s.name, onChange: () => guard(api.renameSession(s.id, nameIn.value), "rename") });
  const isCurrent = state.sessionId === s.id;
  return h(
    "tr",
    { style: isCurrent ? { outline: "1px solid rgba(245,166,35,0.4)" } : undefined },
    h("td", { style: { minWidth: "180px" } }, nameIn, h("div", { class: "small muted" }, fmtDate(s.created_at))),
    h("td", null, h("div", { class: "path" }, s.root_path), h("div", { class: "small muted" }, s.format_filter.length ? s.format_filter.join(", ") : "all formats")),
    h("td", { class: "num" }, `${fmtInt(s.files_selected)} / ${fmtInt(s.files_requested)}`),
    h("td", { class: "num" }, `${s.samples_per_file} × ${s.lines_per_sample}L`),
    h("td", { class: "num" }, String(s.seed)),
    h("td", null,
      s.status === "ready" ? h("span", { class: "pill good" }, "ready") : s.status === "error" ? h("span", { class: "pill bad" }, "error") : h("span", { class: "pill accent" }, s.status),
      s.error ? h("div", { class: "small muted", title: s.error }, s.error.slice(0, 80)) : null,
    ),
    h("td", { style: { minWidth: "200px" } },
      statusBar(s.counts),
      h("div", { class: "small muted" }, `${fmtInt(decided)} / ${fmtInt(s.samples_total)} decided · `, h("span", { style: { color: "var(--good)" } }, `${s.counts.good} good`), " · ", h("span", { style: { color: "var(--bad)" } }, `${s.counts.bad} bad`), " · ", h("span", { style: { color: "var(--skip)" } }, `${s.counts.skip} skipped`)),
    ),
    h("td", null,
      h("div", { class: "row" },
        h("button", { class: "sm primary", disabled: s.status !== "ready", onClick: () => { setSession(s.id); navigate("review", { sessionId: s.id, mode: "pending" }); } }, "Review"),
        h("button", { class: "sm danger", onClick: async () => { if (!confirmDialog(`Delete session "${s.name}" and its ${s.samples_total} samples and annotations?`)) return; await guard(api.deleteSession(s.id), "delete"); if (state.sessionId === s.id) setSession(null); await refreshList(); } }, "Delete"),
      ),
    ),
  );
}

async function refreshList() {
  const sessions = (await guard(api.listSessions(), "sessions")) ?? [];
  clear(listEl);
  if (!sessions.length) {
    listEl.appendChild(h("div", { class: "empty" }, "No sessions yet."));
    return;
  }
  listEl.appendChild(
    h("table", null,
      h("thead", null, h("tr", null, h("th", null, "session"), h("th", null, "corpus"), h("th", { class: "num" }, "files"), h("th", { class: "num" }, "per file"), h("th", { class: "num" }, "seed"), h("th", null, "status"), h("th", null, "progress"), h("th", null, ""))),
      h("tbody", null, ...sessions.map(sessionRow)),
    ),
  );
}

export const sessionsView: View = {
  async render(el, params) {
    roots = (await guard(api.listRoots(), "roots")) ?? [];
    if (params?.rootId) selectedRoot = params.rootId;
    else if (!selectedRoot && roots.length === 1) selectedRoot = roots[0].id;
    if (selectedRoot && !roots.some((r) => r.id === selectedRoot)) selectedRoot = null;
    formEl = h("div", { class: "card" });
    listEl = h("div");
    el.append(
      h("h1", null, "Sessions"),
      h("p", { class: "lead" }, "A session opens a fixed number of files and cuts a fixed number of samples from each. Row-based formats (parquet, jsonl, csv) yield one record per sample; everything else yields a window of N lines. Selection is seeded, so a session can be rebuilt identically."),
      formEl,
      h("h2", null, "All sessions"),
      listEl,
    );
    await loadFormats();
    await renderForm();
    await refreshList();
    unsub = on("job", (j) => {
      if (j.kind === "session") refreshList();
      if (j.kind === "scan" && j.status !== "running") loadFormats().then(renderForm);
    });
  },
  destroy() {
    unsub?.();
    unsub = null;
  },
};
