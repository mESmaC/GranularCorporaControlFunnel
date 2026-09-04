import { save } from "@tauri-apps/plugin-dialog";
import { api, type SessionInfo, type ClassifierRunInfo, type ModelInfo, type EvalReport, type HeadMetrics, type JobProgress } from "../api";
import { h, clear, fmtInt, fmtBytes, fmtDate, fmtPct, toast, guard, on, field, progressBar } from "../ui";
import { state, navigate, type View } from "../state";

let sessions: SessionInfo[] = [];
let models: ModelInfo[] = [];
let runs: ClassifierRunInfo[] = [];
let unsub: (() => void) | null = null;

let trainCard: HTMLElement;
let trainLogEl: HTMLElement;
let modelsEl: HTMLElement;
let runCard: HTMLElement;
let runsEl: HTMLElement;
let reportEl: HTMLElement;
let reportRunId: number | null = null;
let activeTrainJob: JobProgress | null = null;

const LS_TRAIN = "gccf.train.form";
const LS_RUN = "gccf.run.form";

interface TrainForm {
  trainer: "builtin" | "external";
  name: string;
  sessions: number[];
  human_only: boolean;
  parent: number | null;
  feedback_weight: number;
  epochs: number;
  holdout: number;
  min_tag_count: number;
  command_template: string;
  cwd: string;
  worker_template: string;
  model_ext: string;
}

interface RunForm {
  backend: string; // "model:<id>" | "heuristic" | "command"
  mode: "score" | "eval";
  session: string;
  only_pending: boolean;
  workers: number;
  batch_size: number;
  command: string;
  cwd: string;
}

function defaultTrain(): TrainForm {
  return {
    trainer: "builtin",
    name: "",
    sessions: [],
    human_only: true,
    parent: null,
    feedback_weight: 3,
    epochs: 20,
    holdout: 0.2,
    min_tag_count: 5,
    command_template: "py -3 classifier/train_classifier.py {data} --out {model} --parent {parent_model}",
    cwd: "",
    worker_template: "py -3 classifier/worker.py --model {model}",
    model_ext: "joblib",
  };
}
function defaultRun(): RunForm {
  return { backend: "", mode: "eval", session: "", only_pending: true, workers: 4, batch_size: 16, command: "py -3 classifier/worker.py --model classifier/models/gccf-clf.joblib", cwd: "" };
}
function load<T>(key: string, d: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (raw) return { ...d, ...JSON.parse(raw) };
  } catch {
    /* ignore */
  }
  return d;
}
function persist(key: string, v: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(v));
  } catch {
    /* ignore */
  }
}

const tf = load(LS_TRAIN, defaultTrain());
const rf = load(LS_RUN, defaultRun());

function pct(v: number | null | undefined): string {
  return v === null || v === undefined ? "–" : `${(100 * v).toFixed(1)}%`;
}

function headLine(m: HeadMetrics | undefined | null): string {
  if (!m || !m.n_holdout) return "no hold-out";
  return `acc ${pct(m.accuracy)} · P ${pct(m.precision)} · R ${pct(m.recall)} · F1 ${pct(m.f1)} (n=${m.n_holdout})`;
}

function sessionMulti(selected: number[], onChange: (ids: number[]) => void) {
  const sel = h("select", { multiple: true, style: { minHeight: "84px", width: "100%" }, onChange: () => onChange([...sel.selectedOptions].map((o) => Number(o.value))) },
    ...sessions.map((s) => h("option", { value: String(s.id), selected: selected.includes(s.id) }, `${s.name} — ${s.counts.good}✓ ${s.counts.bad}✗ ${s.counts.skip}⏭ ${s.counts.pending}·`)),
  );
  return sel;
}

// ---------------------------------------------------------------------------
// Pipeline banner
// ---------------------------------------------------------------------------

function pipelineBanner(): HTMLElement {
  const labelled = sessions.reduce((a, s) => a + s.counts.good + s.counts.bad, 0);
  const pending = sessions.reduce((a, s) => a + s.counts.pending, 0);
  const latestEval = models.map((m) => m.latest_eval).filter((e): e is NonNullable<typeof e> => !!e)[0];
  const step = (n: string, title: string, detail: string, done: boolean) =>
    h("div", { class: `tile ${done ? "good" : ""}` }, h("div", { class: "v", style: { fontSize: "15px" } }, `${n} · ${title}`), h("div", { class: "l", style: { textTransform: "none", letterSpacing: "0" } }, detail));
  return h("div", { class: "stat-tiles", style: { marginBottom: "14px" } },
    step("1", "Annotate", `${fmtInt(labelled)} human decisions · ${fmtInt(pending)} pending`, labelled >= 20),
    step("2", "Train", `${fmtInt(models.length)} model${models.length === 1 ? "" : "s"}`, models.length > 0),
    step("3", "Evaluate & finetune", latestEval ? `${fmtInt(latestEval.evaluated)} evaluated · ${latestEval.evaluated ? fmtPct(latestEval.agree, latestEval.evaluated) : "–"} agree` : "no evaluation yet", !!latestEval && latestEval.evaluated > 0),
    step("4", "Run", `${fmtInt(runs.filter((r) => r.mode === "score").length)} scoring runs`, runs.some((r) => r.mode === "score" && r.model_row_id !== null)),
  );
}

// ---------------------------------------------------------------------------
// Train
// ---------------------------------------------------------------------------

function renderTrain() {
  clear(trainCard);
  const nameIn = h("input", { type: "text", value: tf.name, placeholder: "e.g. quality-v1", style: { width: "220px" }, onChange: () => { tf.name = nameIn.value; persist(LS_TRAIN, tf); } });
  const trainerSel = h("select", { onChange: () => { tf.trainer = trainerSel.value as any; persist(LS_TRAIN, tf); renderTrain(); } },
    h("option", { value: "builtin", selected: tf.trainer === "builtin" }, "Built-in (Rust · hashed n-gram logistic regression · no Python)"),
    h("option", { value: "external", selected: tf.trainer === "external" }, "External command (your trainer: sklearn / torch / anything)"),
  );
  const parentSel = h("select", { onChange: () => { tf.parent = Number(parentSel.value) || null; persist(LS_TRAIN, tf); renderTrain(); } },
    h("option", { value: "" }, "— none: train from scratch —"),
    ...models.filter((m) => tf.trainer === "external" || m.kind === "builtin").map((m) => h("option", { value: String(m.id), selected: tf.parent === m.id }, `#${m.id} ${m.name}${m.latest_eval ? ` (eval ${m.latest_eval.evaluated ? fmtPct(m.latest_eval.agree, m.latest_eval.evaluated) : "–"})` : ""}`)),
  );
  const sessSel = sessionMulti(tf.sessions, (ids) => { tf.sessions = ids; persist(LS_TRAIN, tf); });
  const humanIn = h("input", { type: "checkbox", checked: tf.human_only, onChange: () => { tf.human_only = humanIn.checked; persist(LS_TRAIN, tf); } });
  const fwIn = h("input", { type: "number", min: 1, max: 20, step: 0.5, value: String(tf.feedback_weight), onChange: () => { tf.feedback_weight = Number(fwIn.value); persist(LS_TRAIN, tf); } });
  const epochsIn = h("input", { type: "number", min: 1, max: 200, value: String(tf.epochs), onChange: () => { tf.epochs = Number(epochsIn.value); persist(LS_TRAIN, tf); } });
  const holdIn = h("input", { type: "number", min: 0, max: 0.5, step: 0.05, value: String(tf.holdout), onChange: () => { tf.holdout = Number(holdIn.value); persist(LS_TRAIN, tf); } });
  const minTagIn = h("input", { type: "number", min: 2, value: String(tf.min_tag_count), onChange: () => { tf.min_tag_count = Number(minTagIn.value); persist(LS_TRAIN, tf); } });
  const cmdIn = h("input", { type: "text", value: tf.command_template, style: { width: "100%" }, onChange: () => { tf.command_template = cmdIn.value; persist(LS_TRAIN, tf); } });
  const cwdIn = h("input", { type: "text", value: tf.cwd, placeholder: "repo root (where classifier/ lives)", style: { width: "100%" }, onChange: () => { tf.cwd = cwdIn.value; persist(LS_TRAIN, tf); } });
  const workerIn = h("input", { type: "text", value: tf.worker_template, style: { width: "100%" }, onChange: () => { tf.worker_template = workerIn.value; persist(LS_TRAIN, tf); } });
  const extIn = h("input", { type: "text", value: tf.model_ext, style: { width: "90px" }, onChange: () => { tf.model_ext = extIn.value; persist(LS_TRAIN, tf); } });

  const start = async () => {
    const parent = tf.parent;
    const res = await guard(api.trainModel(
      tf.trainer === "builtin"
        ? { trainer: "builtin", builtin: { name: tf.name, session_ids: tf.sessions, human_only: tf.human_only, holdout: tf.holdout, min_tag_count: tf.min_tag_count, epochs: tf.epochs, parent_model: parent, feedback_weight: tf.feedback_weight, seed: 42, notes: null } }
        : { trainer: "external", external: { name: tf.name, session_ids: tf.sessions, human_only: tf.human_only, command_template: tf.command_template, cwd: tf.cwd || null, worker_template: tf.worker_template, model_ext: tf.model_ext, parent_model: parent, notes: null } },
    ), "train");
    if (res) {
      toast(parent ? "Finetuning started" : "Training started", "ok");
      trainLogEl.textContent = "starting…";
    }
  };

  const labelled = sessions.reduce((a, s) => a + s.counts.good + s.counts.bad, 0);
  trainCard.append(
    h("div", { class: "card-title" }, "2 · Train", tf.parent ? h("span", { class: "pill accent" }, `finetune from #${tf.parent}`) : null),
    h("p", { class: "small muted", style: { marginTop: 0 } }, "Trains on every human good/bad decision and tag in the chosen sessions. Choosing a base model finetunes it: the built-in trainer warm-starts from its weights and keeps its feature space; samples the base model got wrong during evaluation are up-weighted. The external trainer receives the base model path as {parent_model}."),
    h("div", { class: "stack" },
      h("div", { class: "form-grid" }, field("Model name", nameIn), field("Trainer", trainerSel), field("Base model (finetune)", parentSel), field("Feedback weight", fwIn, "× weight for samples with evaluation disagreements")),
      h("div", { class: "form-grid" },
        field("Sessions (none selected = all)", sessSel),
        h("div", { class: "stack", style: { gap: "6px", justifyContent: "end" } },
          h("label", { class: "check" }, humanIn, "Human decisions only (ignore classifier-applied verdicts)"),
          h("div", { class: "small muted" }, `${fmtInt(labelled)} labelled samples available`),
        ),
      ),
      tf.trainer === "builtin"
        ? h("div", { class: "form-grid" }, field("Epochs", epochsIn, tf.parent ? "fewer for finetuning (5–10)" : "20 is plenty"), field("Hold-out fraction", holdIn), field("Min tag count", minTagIn, "tags seen fewer times are not learned"))
        : h("div", { class: "stack" },
            field("Training command ({data} {model} {parent_model} are substituted)", cmdIn, "also exported as GCCF_DATA / GCCF_MODEL_OUT / GCCF_PARENT_MODEL env vars"),
            h("div", { class: "form-grid" }, field("Working directory", cwdIn), field("Model file extension", extIn)),
            field("Worker command to serve the result ({model} is substituted)", workerIn),
          ),
      h("div", { class: "row" },
        h("button", { class: "primary", disabled: labelled < 20 || (activeTrainJob?.status === "running"), onClick: start }, tf.parent ? "▶ Finetune" : "▶ Train"),
        labelled < 20 ? h("span", { class: "small muted" }, "need at least 20 good/bad decisions") : null,
      ),
      trainLogEl,
    ),
  );
}

function renderTrainLog(j: JobProgress | null) {
  clear(trainLogEl);
  if (!j) return;
  const lines: string[] = Array.isArray(j.payload?.log) ? j.payload.log : [];
  const metrics = j.payload?.metrics;
  trainLogEl.appendChild(h("div", { style: { display: "contents" } },
    h("div", { class: "row" }, h("span", { class: `pill ${j.status === "done" ? "good" : j.status === "error" ? "bad" : "accent"}` }, j.status), h("span", { class: "small" }, j.message), j.status === "running" && j.total > 0 ? h("div", { style: { width: "200px" } }, progressBar(j.done, j.total)) : null),
    lines.length ? h("pre", { class: "code", style: { maxHeight: "220px" } }, lines.slice(-60).join("\n")) : null,
    metrics && metrics.verdict ? h("div", { class: "small" }, h("b", null, "good/bad hold-out: "), headLine(metrics.verdict), metrics.tags?.length ? h("div", null, ...metrics.tags.map((t: any) => h("div", { class: "muted" }, `tag ${t.name}: ${headLine(t.metrics)}`))) : null) : null,
  ));
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

function modelRow(m: ModelInfo): HTMLElement {
  const v = m.metrics?.verdict;
  const ev = m.latest_eval;
  return h("tr", null,
    h("td", null, h("div", null, h("b", null, `#${m.id} ${m.name}`)), h("div", { class: "small muted" }, `${m.kind}${m.parent_id ? ` · finetuned from #${m.parent_id} ${m.parent_name ?? ""}` : ""} · ${fmtDate(m.created_at)}`), h("div", { class: "path", style: { fontSize: "11px" } }, m.path)),
    h("td", { class: "num" }, fmtInt(m.train_rows), m.metrics?.feedback_rows ? h("div", { class: "small muted" }, `${m.metrics.feedback_rows} feedback`) : null),
    h("td", null, h("div", { class: "small" }, headLine(v)), m.tag_names.length ? h("div", { class: "small muted" }, `${m.tag_names.length} tags: ${m.tag_names.slice(0, 6).join(", ")}${m.tag_names.length > 6 ? "…" : ""}`) : h("div", { class: "small muted" }, "no tag heads")),
    h("td", null,
      ev
        ? h("div", null,
            h("div", null, h("b", { style: { color: "var(--good)" } }, ev.evaluated ? fmtPct(ev.agree, ev.evaluated) : "–"), h("span", { class: "small muted" }, ` on ${fmtInt(ev.evaluated)} · ${fmtInt(ev.pending)} left`)),
            h("div", { class: "row", style: { gap: "4px" } },
              ev.pending > 0 ? h("button", { class: "sm primary", onClick: () => navigate("review", { mode: "eval", runId: ev.run_id }) }, "Continue evaluating") : null,
              h("button", { class: "sm", onClick: () => showReport(ev.run_id) }, "Report"),
            ))
        : h("span", { class: "small muted" }, "not evaluated"),
    ),
    h("td", null,
      h("div", { class: "row", style: { gap: "4px" } },
        h("button", { class: "sm", title: "Score fresh pending samples and review them blind", onClick: () => { rf.backend = `model:${m.id}`; rf.mode = "eval"; persist(LS_RUN, rf); renderRun(); runCard.scrollIntoView({ behavior: "smooth" }); } }, "Evaluate…"),
        h("button", { class: "sm", title: "Prefill the trainer to finetune from this model", onClick: () => { tf.parent = m.id; if (m.kind === "external") tf.trainer = "external"; persist(LS_TRAIN, tf); renderTrain(); trainCard.scrollIntoView({ behavior: "smooth" }); } }, "Finetune"),
        h("button", { class: "sm", title: "Score samples for hints / bulk apply", onClick: () => { rf.backend = `model:${m.id}`; rf.mode = "score"; persist(LS_RUN, rf); renderRun(); runCard.scrollIntoView({ behavior: "smooth" }); } }, "Score…"),
        h("button", { class: "sm danger", onClick: async () => { if (!confirm(`Delete model #${m.id} ${m.name}? Its runs and predictions stay; the model file is removed if GCCF wrote it.`)) return; await guard(api.deleteModel(m.id, true), "delete"); await refreshAll(); } }, "✕"),
      ),
    ),
  );
}

function renderModels() {
  clear(modelsEl);
  if (!models.length) {
    modelsEl.appendChild(h("div", { class: "empty" }, "No models yet. Train one above once you have at least 20 good/bad decisions."));
    return;
  }
  modelsEl.appendChild(h("table", null,
    h("thead", null, h("tr", null, h("th", null, "model"), h("th", { class: "num" }, "rows"), h("th", null, "hold-out (training split)"), h("th", null, "evaluation (blind, real data)"), h("th", null, ""))),
    h("tbody", null, ...models.map(modelRow)),
  ));
}

// ---------------------------------------------------------------------------
// Run (score / evaluate)
// ---------------------------------------------------------------------------

function renderRun() {
  clear(runCard);
  const backendSel = h("select", { onChange: () => { rf.backend = backendSel.value; persist(LS_RUN, rf); renderRun(); } },
    h("option", { value: "" }, "— choose —"),
    ...models.map((m) => h("option", { value: `model:${m.id}`, selected: rf.backend === `model:${m.id}` }, `model #${m.id} ${m.name} (${m.kind})`)),
    h("option", { value: "heuristic", selected: rf.backend === "heuristic" }, "built-in heuristic baseline (no model)"),
    h("option", { value: "command", selected: rf.backend === "command" }, "custom worker command"),
  );
  const modeSel = h("select", { onChange: () => { rf.mode = modeSel.value as any; persist(LS_RUN, rf); renderRun(); } },
    h("option", { value: "eval", selected: rf.mode === "eval" }, "Evaluate — score pending samples, then you decide blind; nothing is applied"),
    h("option", { value: "score", selected: rf.mode === "score" }, "Score — predictions as hints, bulk-apply above a threshold"),
  );
  const sessSel = h("select", { onChange: () => { rf.session = sessSel.value; persist(LS_RUN, rf); } },
    h("option", { value: "" }, "all sessions"),
    ...sessions.map((s) => h("option", { value: String(s.id), selected: rf.session === String(s.id) }, `${s.name} — ${s.counts.pending} pending`)),
  );
  const pendingIn = h("input", { type: "checkbox", checked: rf.only_pending || rf.mode === "eval", disabled: rf.mode === "eval", onChange: () => { rf.only_pending = pendingIn.checked; persist(LS_RUN, rf); } });
  const workersIn = h("input", { type: "number", min: 1, max: state.info?.max_workers ?? 8, value: String(rf.workers), onChange: () => { rf.workers = Number(workersIn.value); persist(LS_RUN, rf); } });
  const batchIn = h("input", { type: "number", min: 1, max: 512, value: String(rf.batch_size), onChange: () => { rf.batch_size = Number(batchIn.value); persist(LS_RUN, rf); } });
  const cmdIn = h("input", { type: "text", value: rf.command, style: { width: "100%" }, onChange: () => { rf.command = cmdIn.value; persist(LS_RUN, rf); } });
  const cwdIn = h("input", { type: "text", value: rf.cwd, style: { width: "100%" }, placeholder: "working directory", onChange: () => { rf.cwd = cwdIn.value; persist(LS_RUN, rf); } });

  const start = async () => {
    if (!rf.backend) return toast("Choose a model or backend", "err");
    const isModel = rf.backend.startsWith("model:");
    const res = await guard(api.startClassifierRun({
      session_id: rf.session ? Number(rf.session) : null,
      only_pending: rf.mode === "eval" ? true : rf.only_pending,
      backend: isModel ? "model" : (rf.backend as any),
      model: isModel ? Number(rf.backend.slice(6)) : null,
      command: rf.command,
      args: [],
      cwd: rf.cwd || null,
      model_id: null,
      workers: rf.workers,
      batch_size: rf.batch_size,
      mode: rf.mode,
    }), "start run");
    if (res) {
      toast(rf.mode === "eval" ? "Evaluation run started — review it blind when scoring finishes" : "Scoring run started", "ok");
      await refreshRuns();
    }
  };

  runCard.append(
    h("div", { class: "card-title" }, rf.mode === "eval" ? "3 · Evaluate on real data" : "4 · Run"),
    h("p", { class: "small muted", style: { marginTop: 0 } }, rf.mode === "eval"
      ? "The model scores pending (never-seen) samples read-only. You then review them in the usual card, blind by default; after each decision GCCF reveals what the model said. Your decisions are real annotations, and the disagreements feed the next finetune."
      : "Predictions are stored as hints. Apply them to still-pending samples above a confidence threshold once the model has passed evaluation."),
    h("div", { class: "stack" },
      h("div", { class: "form-grid" }, field("Model / backend", backendSel), field("Mode", modeSel), field("Target", sessSel)),
      rf.backend === "command" ? h("div", { class: "stack" }, field("Worker command", cmdIn), field("Working directory", cwdIn)) : null,
      h("div", { class: "form-grid" }, field("Workers", workersIn, `1–${state.info?.max_workers ?? 8} processes / threads`), field("Batch size", batchIn), h("label", { class: "check", style: { alignSelf: "end" } }, pendingIn, "Only pending samples")),
      h("div", { class: "row" }, h("button", { class: "primary", onClick: start }, rf.mode === "eval" ? "▶ Start evaluation" : "▶ Score")),
    ),
  );
}

// ---------------------------------------------------------------------------
// Runs + evaluation report
// ---------------------------------------------------------------------------

function runRow(r: ClassifierRunInfo): HTMLElement {
  const a = r.agreement;
  const thr = h("input", { type: "number", min: 0, max: 1, step: 0.05, value: "0.8", style: { width: "70px" } });
  const tagsIn = h("input", { type: "checkbox", checked: true });
  const statusPill = h("span", { class: `pill ${r.status === "done" ? "good" : r.status === "error" ? "bad" : r.status === "running" ? "accent" : ""}` }, r.status);
  const scoringDone = r.status !== "running";
  return h("tr", null,
    h("td", null, h("span", { class: `pill ${r.mode === "eval" ? "skip" : ""}` }, r.mode), h("div", null, h("b", null, r.model_id)), h("div", { class: "small muted" }, `#${r.id} · ${r.backend} · ${fmtDate(r.started_at)}`)),
    h("td", null, r.session_name ?? h("span", { class: "muted" }, "all")),
    h("td", null, statusPill, r.status === "running" ? progressBar(r.completed + r.failed, r.total) : null, r.error ? h("div", { class: "small", style: { color: "var(--bad)" }, title: r.error }, r.error.slice(0, 120)) : null),
    h("td", { class: "num" }, `${fmtInt(r.completed)} / ${fmtInt(r.total)}`, r.failed ? h("div", { class: "small", style: { color: "var(--bad)" } }, `${r.failed} failed`) : null),
    h("td", null,
      r.mode === "eval"
        ? h("div", null, h("b", null, r.evaluated ? fmtPct(r.eval_agree, r.evaluated) : "–"), h("span", { class: "small muted" }, ` agreement on ${fmtInt(r.evaluated)} blind decisions`))
        : a && a.compared ? h("div", null, h("div", null, `acc ${fmtPct(a.agree, a.compared)}`), h("div", { class: "small muted" }, `vs. ${a.compared} prior human labels`)) : h("span", { class: "muted small" }, "–"),
    ),
    h("td", null,
      r.mode === "eval"
        ? h("div", { class: "row", style: { gap: "4px" } },
            h("button", { class: "sm primary", disabled: !scoringDone, onClick: () => navigate("review", { mode: "eval", runId: r.id }) }, "Review blind"),
            h("button", { class: "sm", onClick: () => showReport(r.id) }, "Report"),
            h("button", { class: "sm danger", disabled: r.status === "running", onClick: async () => { await guard(api.deleteClassifierRun(r.id), "delete run"); await refreshRuns(); } }, "✕"),
          )
        : h("div", { class: "row", style: { gap: "4px" } },
            h("span", { class: "small muted" }, "≥"), thr,
            h("label", { class: "check small" }, tagsIn, "tags"),
            h("button", { class: "sm", disabled: !scoringDone, onClick: async () => {
              const n = await guard(api.applyRunPredictions(r.id, Number(thr.value), tagsIn.checked), "apply");
              if (n !== undefined) toast(`Applied ${n} prediction(s) to pending samples`, "ok");
              await refreshRuns();
            } }, "Apply to pending"),
            h("button", { class: "sm danger", disabled: r.status === "running", onClick: async () => { await guard(api.deleteClassifierRun(r.id), "delete run"); await refreshRuns(); } }, "✕"),
          ),
    ),
  );
}

function renderRuns() {
  clear(runsEl);
  if (!runs.length) {
    runsEl.appendChild(h("div", { class: "empty" }, "No runs yet."));
    return;
  }
  runsEl.appendChild(h("table", null,
    h("thead", null, h("tr", null, h("th", null, "run"), h("th", null, "target"), h("th", null, "status"), h("th", { class: "num" }, "scored"), h("th", null, "result"), h("th", null, ""))),
    h("tbody", null, ...runs.map(runRow)),
  ));
}

async function showReport(runId: number) {
  reportRunId = runId;
  const rep = await guard(api.evalReport(runId), "report");
  clear(reportEl);
  if (!rep) return;
  reportEl.appendChild(renderReport(rep));
  reportEl.scrollIntoView({ behavior: "smooth", block: "start" });
}

function tile(v: string, l: string, cls = "") {
  return h("div", { class: `tile ${cls}` }, h("div", { class: "v" }, v), h("div", { class: "l" }, l));
}

function renderReport(r: EvalReport): HTMLElement {
  const gate = r.capable
    ? h("div", { class: "status-banner good" }, `Meets the capability gate: accuracy ${pct(r.accuracy)} ≥ ${Math.round(r.gate_min_accuracy * 100)}% on ${fmtInt(r.evaluated)} ≥ ${fmtInt(r.gate_min_evaluated)} blind decisions. Scoring runs of this model can be applied.`)
    : h("div", { class: "status-banner skip" }, `Not yet capable: need accuracy ≥ ${Math.round(r.gate_min_accuracy * 100)}% on at least ${fmtInt(r.gate_min_evaluated)} blind decisions (have ${pct(r.accuracy)} on ${fmtInt(r.evaluated)}). Thresholds live in Settings.`);
  const model = r.model_row_id !== null ? models.find((m) => m.id === r.model_row_id) : undefined;
  return h("div", { class: "card" },
    h("div", { class: "card-title" }, `Evaluation report · run #${r.run_id} · ${r.model_name}`, h("span", { style: { flex: "1" } }), h("button", { class: "ghost sm", onClick: () => { reportRunId = null; clear(reportEl); } }, "close")),
    gate,
    h("div", { class: "stat-tiles", style: { marginTop: "12px" } },
      tile(fmtInt(r.evaluated), "blind decisions"),
      tile(fmtInt(r.remaining), "still to evaluate"),
      tile(pct(r.accuracy), "accuracy", r.capable ? "good" : ""),
      tile(pct(r.precision), "precision (good)"),
      tile(pct(r.recall), "recall (good)"),
      tile(pct(r.f1), "F1 (good)"),
      tile(`${r.blind}/${r.evaluated}`, "decided blind"),
    ),
    h("div", { class: "grid-2", style: { marginTop: "14px" } },
      h("div", null,
        h("h2", { style: { marginTop: 0 } }, "Confusion"),
        h("table", null,
          h("thead", null, h("tr", null, h("th", null, ""), h("th", { class: "num" }, "you: good"), h("th", { class: "num" }, "you: bad"))),
          h("tbody", null,
            h("tr", null, h("td", null, "model: good"), h("td", { class: "num", style: { color: "var(--good)" } }, fmtInt(r.tp)), h("td", { class: "num", style: { color: "var(--bad)" } }, `${fmtInt(r.fp)} (false keep)`)),
            h("tr", null, h("td", null, "model: bad"), h("td", { class: "num", style: { color: "var(--bad)" } }, `${fmtInt(r.fn_)} (false drop)`), h("td", { class: "num", style: { color: "var(--good)" } }, fmtInt(r.tn))),
          )),
        h("h2", null, "Calibration"),
        h("table", null,
          h("thead", null, h("tr", null, h("th", null, "confidence"), h("th", { class: "num" }, "n"), h("th", { class: "num" }, "agreement"))),
          h("tbody", null, ...r.calibration.map((b) => h("tr", null, h("td", null, `${Math.round(b.lo * 100)}–${Math.round(b.hi * 100)}%`), h("td", { class: "num" }, fmtInt(b.n)), h("td", { class: "num" }, b.n ? fmtPct(b.agree, b.n) : "–")))),
        ),
        h("div", { class: "small muted" }, "A well-calibrated model agrees with you about as often as its confidence says. If the 90–100% bucket is far below 90%, don't trust the threshold."),
      ),
      h("div", null,
        h("h2", { style: { marginTop: 0 } }, "Tags"),
        r.tags.length
          ? h("table", null,
              h("thead", null, h("tr", null, h("th", null, "tag"), h("th", { class: "num" }, "model"), h("th", { class: "num" }, "you"), h("th", { class: "num" }, "both"), h("th", { class: "num" }, "P / R"))),
              h("tbody", null, ...r.tags.map((t) => h("tr", null, h("td", null, t.name), h("td", { class: "num" }, fmtInt(t.predicted)), h("td", { class: "num" }, fmtInt(t.human)), h("td", { class: "num" }, fmtInt(t.both)), h("td", { class: "num" }, `${t.predicted ? fmtPct(t.both, t.predicted) : "–"} / ${t.human ? fmtPct(t.both, t.human) : "–"}`)))),
            )
          : h("div", { class: "small muted" }, "no tags involved yet"),
        h("h2", null, "Next step"),
        h("div", { class: "row" },
          model ? h("button", { class: "primary", onClick: () => { tf.parent = model.id; if (model.kind === "external") tf.trainer = "external"; persist(LS_TRAIN, tf); renderTrain(); trainCard.scrollIntoView({ behavior: "smooth" }); } }, `Finetune #${model.id} on this feedback`) : null,
          r.remaining > 0 ? h("button", { onClick: () => navigate("review", { mode: "eval", runId: r.run_id }) }, `Continue evaluating (${fmtInt(r.remaining)} left)`) : null,
          model && r.capable ? h("button", { onClick: () => { rf.backend = `model:${model.id}`; rf.mode = "score"; persist(LS_RUN, rf); renderRun(); runCard.scrollIntoView({ behavior: "smooth" }); } }, "Score new samples") : null,
        ),
        h("div", { class: "small muted", style: { marginTop: "8px" } }, `${fmtInt(r.evaluated - r.agree)} disagreement(s) will be up-weighted when finetuning.`),
      ),
    ),
    r.disagreements.length
      ? h("div", null,
          h("h2", null, `Disagreements (${fmtInt(r.disagreements.length)}, most confident first)`),
          h("table", null,
            h("thead", null, h("tr", null, h("th", null, "model"), h("th", null, "you"), h("th", null, "file"), h("th", null, "preview"), h("th", null, ""))),
            h("tbody", null, ...r.disagreements.slice(0, 60).map((d) => h("tr", null,
              h("td", null, h("span", { class: `pill ${d.predicted === "good" ? "good" : "bad"}` }, `${d.predicted} ${Math.round(d.score * 100)}%`)),
              h("td", null, h("span", { class: `pill ${d.human === "good" ? "good" : "bad"}` }, d.human)),
              h("td", null, h("div", { class: "path" }, d.rel_path)),
              h("td", null, h("div", { class: "mono small", style: { maxHeight: "48px", overflow: "hidden", whiteSpace: "pre-wrap", wordBreak: "break-word", color: "var(--text-2)" } }, d.preview)),
              h("td", null, h("button", { class: "sm", onClick: () => navigate("review", { sessionId: d.session_id, sampleId: d.sample_id, mode: "decided" }) }, "Open")),
            ))),
          ))
      : null,
  );
}

// ---------------------------------------------------------------------------
// Advanced: export + protocol
// ---------------------------------------------------------------------------

function advancedCard(): HTMLElement {
  let selected: number[] = [];
  let includeSkipped = true;
  let includePending = false;
  let humanOnly = true;
  const sel = sessionMulti([], (ids) => { selected = ids; });
  const out = h("div", { class: "small muted" });
  return h("details", { class: "card" },
    h("summary", null, "Advanced: export training data · worker protocol"),
    h("div", { class: "stack", style: { marginTop: "10px" } },
      h("div", { class: "form-grid" },
        field("Sessions (none selected = all)", sel),
        h("div", { class: "stack", style: { gap: "6px", justifyContent: "end" } },
          h("label", { class: "check" }, h("input", { type: "checkbox", checked: humanOnly, onChange: (e: Event) => { humanOnly = (e.target as HTMLInputElement).checked; } }), "Human decisions only"),
          h("label", { class: "check" }, h("input", { type: "checkbox", checked: includeSkipped, onChange: (e: Event) => { includeSkipped = (e.target as HTMLInputElement).checked; } }), "Include skipped (label = null, tags still useful)"),
          h("label", { class: "check" }, h("input", { type: "checkbox", checked: includePending, onChange: (e: Event) => { includePending = (e.target as HTMLInputElement).checked; } }), "Include pending"),
        ),
      ),
      h("div", { class: "row" },
        h("button", { onClick: async () => {
          const path = await save({ title: "Export training data", defaultPath: "gccf-training.jsonl", filters: [{ name: "JSON Lines", extensions: ["jsonl"] }] });
          if (!path) return;
          const st = await guard(api.exportTrainingData(path, selected, includePending, includeSkipped, humanOnly), "export");
          if (st) {
            out.textContent = `${fmtInt(st.rows)} rows (${fmtBytes(st.bytes)}) — ${st.good} good, ${st.bad} bad, ${st.skip} skipped → ${path}`;
            toast("Training data written", "ok");
          }
        } }, "Export JSONL…"),
        out,
      ),
      h("div", { class: "help" },
        h("div", null, "Each exported row carries text, verdict/label, tags, locator, provenance, and — after an evaluation round — an ", h("code", null, "eval"), " object: what the model predicted, its confidence, what you said, and whether they agreed. That is the signal to finetune on."),
        h("div", { style: { marginTop: "6px" } }, "Worker protocol (JSON lines over stdio; a blank line ends a batch both ways; ", h("code", null, "GCCF_WORKER_INDEX"), " identifies the worker):"),
        h("pre", { class: "code" }, `→ {"id": 12, "text": "...", "path": "shard-1.jsonl.zst", "format": "jsonl"}\n→\n← {"id": 12, "verdict": "good", "score": 0.93, "tags": ["english"]}\n←`),
        h("div", null, "Trainer contract: read {data} (the JSONL above), write {model}; optionally print a final line ", h("code", null, "{\"gccf_metrics\": {...}}"), " to have metrics stored with the model. classifier/train_classifier.py and classifier/worker.py implement both."),
      ),
    ),
  );
}

// ---------------------------------------------------------------------------

async function refreshRuns() {
  runs = (await guard(api.listClassifierRuns(), "runs")) ?? [];
  renderRuns();
}

async function refreshAll() {
  [sessions, models, runs] = await Promise.all([
    guard(api.listSessions(), "sessions").then((v) => v ?? []),
    guard(api.listModels(), "models").then((v) => v ?? []),
    guard(api.listClassifierRuns(), "runs").then((v) => v ?? []),
  ]);
  const banner = document.getElementById("clf-banner");
  if (banner) {
    clear(banner);
    banner.appendChild(pipelineBanner());
  }
  renderTrain();
  renderModels();
  renderRun();
  renderRuns();
  if (reportRunId !== null) await showReport(reportRunId);
}

export const classifierView: View = {
  async render(el, params) {
    trainCard = h("div", { class: "card" });
    trainLogEl = h("div", { class: "stack", style: { gap: "6px" } });
    modelsEl = h("div");
    runCard = h("div", { class: "card" });
    runsEl = h("div");
    reportEl = h("div");
    el.append(
      h("h1", null, "Classifier"),
      h("p", { class: "lead" }, "Close the loop: your decisions train a model, the model is evaluated blind on real data you haven't reviewed, the disagreements finetune it, and only a model that clears the gate gets to score at scale."),
      h("div", { id: "clf-banner" }),
      trainCard,
      h("h2", null, "Models"),
      modelsEl,
      runCard,
      h("h2", null, "Runs"),
      runsEl,
      reportEl,
      advancedCard(),
    );
    await refreshAll();
    if (params?.runId) await showReport(params.runId);
    unsub = on("job", (j: JobProgress) => {
      if (j.kind === "train") {
        activeTrainJob = j;
        renderTrainLog(j);
        if (j.status !== "running") refreshAll();
      } else if (j.kind === "classify") {
        refreshRuns();
        if (j.status !== "running") refreshAll();
      }
    });
  },
  destroy() {
    unsub?.();
    unsub = null;
  },
};
