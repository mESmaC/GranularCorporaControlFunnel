import { save } from "@tauri-apps/plugin-dialog";
import { api, type SessionInfo, type ClassifierRunInfo } from "../api";
import { h, clear, fmtInt, fmtBytes, fmtDate, fmtPct, toast, guard, on, field, progressBar } from "../ui";
import { state, type View } from "../state";

let sessions: SessionInfo[] = [];
let runsEl: HTMLElement;
let unsub: (() => void) | null = null;
const LS = "gccf.classifier.form";

interface FormState {
  backend: "heuristic" | "command";
  command: string;
  cwd: string;
  model_id: string;
  workers: number;
  batch_size: number;
  session: string;
  only_pending: boolean;
}

function loadForm(): FormState {
  try {
    const raw = localStorage.getItem(LS);
    if (raw) return { ...defaultForm(), ...JSON.parse(raw) };
  } catch {
    /* ignore */
  }
  return defaultForm();
}
function defaultForm(): FormState {
  return { backend: "heuristic", command: "py -3 classifier/worker.py --model classifier/models/gccf-clf.joblib", cwd: "", model_id: "", workers: 4, batch_size: 16, session: "", only_pending: true };
}
function saveForm(f: FormState) {
  try {
    localStorage.setItem(LS, JSON.stringify(f));
  } catch {
    /* ignore */
  }
}

function sessionSelect(value: string, onChange: (v: string) => void, allowAll = true, multiple = false) {
  const sel = h("select", { multiple, onChange: () => onChange(multiple ? [...sel.selectedOptions].map((o) => o.value).join(",") : sel.value), style: multiple ? { minHeight: "90px" } : undefined });
  if (allowAll && !multiple) sel.appendChild(h("option", { value: "" }, "all sessions"));
  for (const s of sessions) sel.appendChild(h("option", { value: String(s.id), selected: value.split(",").includes(String(s.id)) }, `${s.name} — ${s.counts.good}✓ ${s.counts.bad}✗ ${s.counts.pending} pending`));
  return sel;
}

function exportCard(): HTMLElement {
  let selected: number[] = [];
  let includeSkipped = false;
  let includePending = false;
  let humanOnly = true;
  const sel = sessionSelect("", (v) => { selected = v ? v.split(",").filter(Boolean).map(Number) : []; }, true, true);
  const out = h("div", { class: "small muted" });
  return h("div", { class: "card" },
    h("div", { class: "card-title" }, "1 · Export annotations as training data"),
    h("p", { class: "small muted", style: { marginTop: 0 } }, "Writes one JSON object per line: text, verdict (good/bad), numeric label, tags, locator and provenance. Feed it to classifier/train_classifier.py, or your own trainer."),
    h("div", { class: "form-grid" },
      field("Sessions (none selected = all)", sel),
      h("div", { class: "stack", style: { gap: "6px", justifyContent: "end" } },
        h("label", { class: "check" }, h("input", { type: "checkbox", checked: humanOnly, onChange: (e: Event) => { humanOnly = (e.target as HTMLInputElement).checked; } }), "Human decisions only (drop classifier-applied verdicts)"),
        h("label", { class: "check" }, h("input", { type: "checkbox", checked: includeSkipped, onChange: (e: Event) => { includeSkipped = (e.target as HTMLInputElement).checked; } }), "Include skipped (label = null)"),
        h("label", { class: "check" }, h("input", { type: "checkbox", checked: includePending, onChange: (e: Event) => { includePending = (e.target as HTMLInputElement).checked; } }), "Include pending (unlabelled, for inference tests)"),
      ),
    ),
    h("div", { class: "row", style: { marginTop: "10px" } },
      h("button", { class: "primary", onClick: async () => {
        const path = await save({ title: "Export training data", defaultPath: "gccf-training.jsonl", filters: [{ name: "JSON Lines", extensions: ["jsonl"] }] });
        if (!path) return;
        const st = await guard(api.exportTrainingData(path, selected, includePending, includeSkipped, humanOnly), "export");
        if (st) {
          out.textContent = `${fmtInt(st.rows)} rows (${fmtBytes(st.bytes)}) — ${st.good} good, ${st.bad} bad, ${st.skip} skipped, ${st.pending} pending · ${st.human} human / ${st.classifier} classifier → ${path}`;
          toast("Training data written", "ok");
        }
      } }, "Export JSONL…"),
      out,
    ),
  );
}

function runCard(): HTMLElement {
  const f = loadForm();
  const backendSel = h("select", { onChange: () => { f.backend = backendSel.value as any; saveForm(f); sync(); } },
    h("option", { value: "heuristic", selected: f.backend === "heuristic" }, "Built-in heuristic baseline (no model)"),
    h("option", { value: "command", selected: f.backend === "command" }, "External worker process (your model, GPU)"),
  );
  const cmdIn = h("input", { type: "text", value: f.command, style: { width: "100%" }, onChange: () => { f.command = cmdIn.value; saveForm(f); } });
  const cwdIn = h("input", { type: "text", value: f.cwd, placeholder: "defaults to the GCCF working directory", style: { width: "100%" }, onChange: () => { f.cwd = cwdIn.value; saveForm(f); } });
  const modelIn = h("input", { type: "text", value: f.model_id, placeholder: "auto", onChange: () => { f.model_id = modelIn.value; saveForm(f); } });
  const workersIn = h("input", { type: "number", min: 1, max: state.info?.max_workers ?? 8, value: String(f.workers), onChange: () => { f.workers = Number(workersIn.value); saveForm(f); } });
  const batchIn = h("input", { type: "number", min: 1, max: 512, value: String(f.batch_size), onChange: () => { f.batch_size = Number(batchIn.value); saveForm(f); } });
  const sessSel = sessionSelect(f.session, (v) => { f.session = v; saveForm(f); });
  const pendingIn = h("input", { type: "checkbox", checked: f.only_pending, onChange: () => { f.only_pending = pendingIn.checked; saveForm(f); } });
  const cmdBlock = h("div", { class: "stack" },
    field("Worker command (one process per worker; each receives GCCF_WORKER_INDEX)", cmdIn, "e.g. py -3 classifier/worker.py --model classifier/models/gccf-clf.joblib --device cuda"),
    field("Working directory", cwdIn),
  );
  const sync = () => { cmdBlock.style.display = f.backend === "command" ? "" : "none"; };
  sync();
  return h("div", { class: "card" },
    h("div", { class: "card-title" }, "2 · Run a classifier over samples"),
    h("p", { class: "small muted", style: { marginTop: 0 } }, "Predictions are stored next to the samples, never applied automatically. The reviewer shows the latest prediction as a hint, and you can bulk-apply a run above a confidence threshold to samples that are still pending."),
    h("div", { class: "stack" },
      field("Backend", backendSel),
      cmdBlock,
      h("div", { class: "form-grid" },
        field("Model id (for provenance)", modelIn),
        field("Workers", workersIn, `1–${state.info?.max_workers ?? 8} concurrent worker processes`),
        field("Batch size", batchIn, "samples per request to a worker"),
        field("Target", sessSel),
      ),
      h("label", { class: "check" }, pendingIn, "Only samples that are still pending (uncheck to also score decided samples → agreement stats)"),
      h("div", { class: "row" },
        h("button", { class: "primary", onClick: async () => {
          const res = await guard(api.startClassifierRun({
            session_id: f.session ? Number(f.session) : null,
            only_pending: f.only_pending,
            backend: f.backend,
            command: f.command,
            args: [],
            cwd: f.cwd || null,
            model_id: f.model_id || null,
            workers: f.workers,
            batch_size: f.batch_size,
          }), "start run");
          if (res) toast("Classifier run started", "ok");
          await refreshRuns();
        } }, "▶ Run"),
      ),
    ),
  );
}

function runRow(r: ClassifierRunInfo): HTMLElement {
  const a = r.agreement;
  const acc = a && a.compared ? fmtPct(a.agree, a.compared) : "–";
  const prec = a && a.tp + a.fp ? fmtPct(a.tp, a.tp + a.fp) : "–";
  const rec = a && a.tp + a.fn_ ? fmtPct(a.tp, a.tp + a.fn_) : "–";
  const thr = h("input", { type: "number", min: 0, max: 1, step: 0.05, value: "0.8", style: { width: "70px" } });
  const tagsIn = h("input", { type: "checkbox", checked: true });
  const statusPill = h("span", { class: `pill ${r.status === "done" ? "good" : r.status === "error" ? "bad" : r.status === "running" ? "accent" : ""}` }, r.status);
  return h("tr", null,
    h("td", null, h("div", null, h("b", null, r.model_id)), h("div", { class: "small muted" }, `${r.backend}${r.command ? " · " + r.command : ""}`), h("div", { class: "small muted" }, fmtDate(r.started_at))),
    h("td", null, r.session_name ?? h("span", { class: "muted" }, "all")),
    h("td", null, statusPill, r.status === "running" ? progressBar(r.completed + r.failed, r.total) : null, r.error ? h("div", { class: "small", style: { color: "var(--bad)" }, title: r.error }, r.error.slice(0, 120)) : null),
    h("td", { class: "num" }, `${fmtInt(r.completed)} / ${fmtInt(r.total)}`, r.failed ? h("div", { class: "small", style: { color: "var(--bad)" } }, `${r.failed} failed`) : null),
    h("td", { class: "num" }, a && a.compared ? h("div", null, h("div", null, `acc ${acc}`), h("div", { class: "small muted" }, `P ${prec} · R ${rec} · n=${a.compared}`)) : h("span", { class: "muted small" }, "no overlap with human decisions")),
    h("td", null,
      h("div", { class: "row" },
        h("span", { class: "small muted" }, "≥"), thr,
        h("label", { class: "check small" }, tagsIn, "tags"),
        h("button", { class: "sm", disabled: r.status === "running", onClick: async () => {
          const n = await guard(api.applyRunPredictions(r.id, Number(thr.value), tagsIn.checked), "apply");
          if (n !== undefined) toast(`Applied ${n} prediction(s) to pending samples`, "ok");
          await refreshRuns();
        } }, "Apply to pending"),
        h("button", { class: "sm danger", disabled: r.status === "running", onClick: async () => { await guard(api.deleteClassifierRun(r.id), "delete run"); await refreshRuns(); } }, "✕"),
      ),
    ),
  );
}

async function refreshRuns() {
  const runs = (await guard(api.listClassifierRuns(), "runs")) ?? [];
  clear(runsEl);
  if (!runs.length) {
    runsEl.appendChild(h("div", { class: "empty" }, "No classifier runs yet."));
    return;
  }
  runsEl.appendChild(
    h("table", null,
      h("thead", null, h("tr", null, h("th", null, "model"), h("th", null, "target"), h("th", null, "status"), h("th", { class: "num" }, "scored"), h("th", { class: "num" }, "agreement with humans"), h("th", null, "apply"))),
      h("tbody", null, ...runs.map(runRow)),
    ),
  );
}

export const classifierView: View = {
  async render(el) {
    sessions = (await guard(api.listSessions(), "sessions")) ?? [];
    runsEl = h("div");
    el.append(
      h("h1", null, "Classifier"),
      h("p", { class: "lead" }, "Every decision you make is a training signal. Export them, train a model, then point GCCF at it: up to eight worker processes score samples in parallel, and the reviewer surfaces predictions so the model gradually takes over the easy calls."),
      exportCard(),
      runCard(),
      h("div", { class: "card" },
        h("div", { class: "card-title" }, "Worker protocol"),
        h("div", { class: "help" },
          h("div", null, "GCCF starts your command once per worker and speaks JSON lines over stdin/stdout. A batch is a run of request lines followed by an empty line; reply with one line per item (any order) and an empty line. Items you omit are recorded as failed. stderr is passed through to the GCCF log."),
          h("pre", { class: "code" }, `→ {"id": 12, "text": "...", "path": "shard-1.jsonl.zst", "format": "jsonl"}\n→ {"id": 13, "text": "...", "path": "...", "format": "..."}\n→\n← {"id": 12, "verdict": "good", "score": 0.93, "tags": ["english"]}\n← {"id": 13, "verdict": "bad", "score": 0.71, "tags": []}\n←`),
          h("div", null, "classifier/worker.py implements this for a scikit-learn model trained by classifier/train_classifier.py; swap the model-loading and predict functions for a torch model to use the GPU."),
        ),
      ),
      h("h2", null, "Runs"),
      runsEl,
    );
    await refreshRuns();
    unsub = on("job", (j) => { if (j.kind === "classify") refreshRuns(); });
  },
  destroy() {
    unsub?.();
    unsub = null;
  },
};
