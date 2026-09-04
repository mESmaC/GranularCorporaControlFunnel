import { save } from "@tauri-apps/plugin-dialog";
import { api, type SessionInfo, type ReceiptOptions, type ReceiptSummary } from "../api";
import { h, clear, fmtInt, fmtBytes, toast, guard, field, debounce } from "../ui";
import type { View } from "../state";

let sessions: SessionInfo[] = [];
let previewEl: HTMLElement;
const LS = "gccf.receipt.options";

function loadOptions(): ReceiptOptions {
  const d: ReceiptOptions = {
    session_ids: [],
    verbosity: "files",
    include_min_good_ratio: 0.8,
    exclude_max_good_ratio: 0.5,
    min_decided: 1,
    include_unsampled_files: false,
    skip_counts_as_bad: false,
    include_classifier_decisions: true,
    include_text: false,
    note: null,
  };
  try {
    const raw = localStorage.getItem(LS);
    if (raw) return { ...d, ...JSON.parse(raw), session_ids: [] };
  } catch {
    /* ignore */
  }
  return d;
}

function tile(v: string, l: string, cls = "") {
  return h("div", { class: `tile ${cls}` }, h("div", { class: "v" }, v), h("div", { class: "l" }, l));
}

function renderPreview(s: ReceiptSummary) {
  clear(previewEl);
  const tags = Object.entries(s.tags).sort((a, b) => b[1] - a[1]);
  previewEl.appendChild(h("div", null,
    h("div", { class: "stat-tiles" },
      tile(fmtInt(s.files_include), "files include", "good"),
      tile(fmtInt(s.files_exclude), "files exclude", "bad"),
      tile(fmtInt(s.files_review), "files need review", "skip"),
      tile(fmtInt(s.files_unsampled), "unsampled listed"),
      tile(fmtBytes(s.bytes_include), "bytes included", "good"),
      tile(fmtBytes(s.bytes_exclude), "bytes excluded", "bad"),
      tile(fmtInt(s.exclusions), "record exclusions", "accent"),
    ),
    h("div", { class: "stat-tiles", style: { marginTop: "10px" } },
      tile(fmtInt(s.samples.samples), "samples"),
      tile(fmtInt(s.samples.good), "good", "good"),
      tile(fmtInt(s.samples.bad), "bad", "bad"),
      tile(fmtInt(s.samples.skip), "skipped", "skip"),
      tile(fmtInt(s.samples.pending), "pending"),
      tile(`${fmtInt(s.samples.human)} / ${fmtInt(s.samples.classifier)}`, "human / classifier"),
    ),
    tags.length ? h("div", { class: "chips", style: { marginTop: "10px" } }, ...tags.map(([n, c]) => h("span", { class: "chip" }, n, h("span", { class: "cnt" }, String(c))))) : null,
  ));
}

export const receiptView: View = {
  async render(el) {
    sessions = (await guard(api.listSessions(), "sessions")) ?? [];
    const opts = loadOptions();
    const persist = () => { try { localStorage.setItem(LS, JSON.stringify(opts)); } catch { /* ignore */ } };

    const refresh = debounce(async () => {
      const s = await guard(api.receiptPreview(opts), "preview");
      if (s) renderPreview(s);
    }, 200);

    const sessSel = h("select", { multiple: true, style: { minHeight: "110px", width: "100%" }, onChange: () => { opts.session_ids = [...sessSel.selectedOptions].map((o) => Number(o.value)); refresh(); } },
      ...sessions.map((s) => h("option", { value: String(s.id) }, `${s.name} — ${s.counts.good}✓ ${s.counts.bad}✗ ${s.counts.skip}⏭ ${s.counts.pending}·`)),
    );
    const verb = h("select", { onChange: () => { opts.verbosity = verb.value as any; persist(); refresh(); } },
      h("option", { value: "summary", selected: opts.verbosity === "summary" }, "summary — totals only"),
      h("option", { value: "files", selected: opts.verbosity === "files" }, "files — per-file verdicts, tag counts, record exclusions"),
      h("option", { value: "full", selected: opts.verbosity === "full" }, "full — files plus every sample with locator and tags"),
    );
    const num = (key: keyof ReceiptOptions, step: number, min: number, max: number) => {
      const i = h("input", { type: "number", step, min, max, value: String(opts[key]), onChange: () => { (opts as any)[key] = Number(i.value); persist(); refresh(); } });
      return i;
    };
    const chk = (key: keyof ReceiptOptions, label: string, hint?: string) =>
      h("label", { class: "check", title: hint ?? "" }, h("input", { type: "checkbox", checked: Boolean(opts[key]), onChange: (e: Event) => { (opts as any)[key] = (e.target as HTMLInputElement).checked; persist(); refresh(); } }), label);
    const noteIn = h("input", { type: "text", placeholder: "free-text note stored in the receipt", style: { width: "100%" }, value: opts.note ?? "", onChange: () => { opts.note = noteIn.value || null; persist(); } });
    const out = h("div", { class: "small muted" });

    previewEl = h("div");
    el.append(
      h("h1", null, "Ingestion receipt"),
      h("p", { class: "lead" }, "The receipt is the output of all this work: a versioned JSON document your training-prep tool reads to decide which files to ingest, which records inside them to drop, and how each was tagged. Nothing is copied; every entry points back into the corpus by path and locator."),
      h("div", { class: "grid-2" },
        h("div", { class: "card" },
          h("div", { class: "card-title" }, "Scope & policy"),
          h("div", { class: "stack" },
            field("Sessions (none selected = all)", sessSel),
            field("Verbosity", verb),
            h("div", { class: "form-grid" },
              field("Include if good ratio ≥", num("include_min_good_ratio", 0.05, 0, 1)),
              field("Exclude if good ratio ≤", num("exclude_max_good_ratio", 0.05, 0, 1)),
              field("Min decided samples per file", num("min_decided", 1, 0, 10000), "files below this get verdict \"review\""),
            ),
            chk("skip_counts_as_bad", "Treat skipped samples as bad"),
            chk("include_classifier_decisions", "Count classifier-applied verdicts (uncheck for human-only receipts)"),
            chk("include_unsampled_files", "List files that were never sampled (verdict \"unsampled\")"),
            chk("include_text", "Embed sample text (full verbosity only; large)"),
            field("Note", noteIn),
          ),
        ),
        h("div", { class: "card" },
          h("div", { class: "card-title" }, "Preview"),
          previewEl,
          h("div", { class: "row", style: { marginTop: "14px" } },
            h("button", { class: "primary", onClick: async () => {
              const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
              const path = await save({ title: "Write ingestion receipt", defaultPath: `gccf-receipt-${stamp}.json`, filters: [{ name: "JSON", extensions: ["json"] }] });
              if (!path) return;
              const s = await guard(api.exportReceipt(path, opts), "write receipt");
              if (s) {
                renderPreview(s);
                out.textContent = `written → ${path}`;
                toast("Receipt written", "ok");
                out.appendChild(h("button", { class: "ghost sm", onClick: () => guard(api.revealPath(path), "reveal") }, "reveal"));
              }
            } }, "Write receipt…"),
            out,
          ),
          h("div", { class: "help", style: { marginTop: "14px" } }, "Schema: docs/receipt-schema.md. File verdicts are include / exclude / review / unsampled; each file also carries the locators of its bad records so an ingester can drop rows without dropping the file."),
        ),
      ),
    );
    refresh();
  },
};
