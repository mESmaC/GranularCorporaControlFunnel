import { api } from "../api";
import { h, toast, guard, field } from "../ui";
import { state, refreshSettings, type View } from "../state";

export const settingsView: View = {
  async render(el) {
    await refreshSettings();
    const s = state.settings;
    const threads = h("input", { type: "number", min: 1, max: 64, value: s["sample_threads"] ?? String(state.info?.default_threads ?? 4) });
    const maxKb = h("input", { type: "number", min: 1, value: String(Math.round(Number(s["max_bytes_per_sample"] ?? 262144) / 1024)) });
    const streamGb = h("input", { type: "number", min: 0, step: 0.5, value: String(Number(s["max_stream_bytes"] ?? 0) / 1e9) });
    const gateAcc = h("input", { type: "number", min: 0.5, max: 1, step: 0.01, value: s["eval_min_accuracy"] ?? "0.9" });
    const gateN = h("input", { type: "number", min: 1, value: s["eval_min_evaluated"] ?? "100" });
    const saveAll = async () => {
      await guard(api.setSetting("sample_threads", String(Math.max(1, Number(threads.value)))));
      await guard(api.setSetting("max_bytes_per_sample", String(Math.max(1024, Number(maxKb.value) * 1024))));
      await guard(api.setSetting("max_stream_bytes", String(Math.round(Number(streamGb.value) * 1e9))));
      await guard(api.setSetting("eval_min_accuracy", String(Math.min(1, Math.max(0, Number(gateAcc.value))))));
      await guard(api.setSetting("eval_min_evaluated", String(Math.max(1, Math.round(Number(gateN.value))))));
      await refreshSettings();
      if (state.info) state.info.default_threads = Number(threads.value);
      toast("Settings saved", "ok", 1500);
    };
    el.append(
      h("h1", null, "Settings"),
      h("div", { class: "card" },
        h("div", { class: "card-title" }, "Sampling defaults"),
        h("div", { class: "form-grid" },
          field("Sampling threads", threads, `files sampled concurrently (${state.info?.cpus ?? "?"} cores available)`),
          field("Max KB kept per sample", maxKb, "longer windows are cut and flagged truncated"),
          field("Compressed stream cap (GB)", streamGb, "stop reading a .gz/.zst file after this much decompressed data (0 = whole file)"),
        ),
        h("div", { class: "row", style: { marginTop: "12px" } }, h("button", { class: "primary", onClick: saveAll }, "Save")),
      ),
      h("div", { class: "card" },
        h("div", { class: "card-title" }, "Classifier capability gate"),
        h("p", { class: "small muted", style: { marginTop: 0 } }, "An evaluation run counts as passing when the model agrees with your blind decisions at least this often, over at least this many decisions. The gate is advisory: it labels reports and models, it does not block anything."),
        h("div", { class: "form-grid" },
          field("Minimum accuracy", gateAcc, "0.9 = agree on 90% of blind decisions"),
          field("Minimum blind decisions", gateN, "small n gives noisy accuracy"),
        ),
        h("div", { class: "row", style: { marginTop: "12px" } }, h("button", { class: "primary", onClick: saveAll }, "Save")),
      ),
      h("div", { class: "card" },
        h("div", { class: "card-title" }, "Catalogue"),
        h("div", { class: "kv" },
          h("span", { class: "k" }, "database"), h("span", { class: "path" }, state.info?.db_path ?? "–"),
          h("span", { class: "k" }, "version"), h("span", null, state.info?.version ?? "–"),
          h("span", { class: "k" }, "override"), h("span", { class: "small muted" }, "set the GCCF_DB environment variable to a file path to use a different catalogue (e.g. one per project)"),
        ),
        h("div", { class: "row", style: { marginTop: "12px" } },
          h("button", { onClick: () => state.info && guard(api.revealPath(state.info.db_path), "reveal") }, "Reveal database"),
        ),
        h("div", { class: "help", style: { marginTop: "12px" } }, "The catalogue is the only state GCCF owns: file index, samples, every decision and tag, and classifier predictions. Back it up like source code. Corpora on disk are never modified."),
      ),
      h("div", { class: "card" },
        h("div", { class: "card-title" }, "Keyboard"),
        h("div", { class: "kv" },
          h("span", { class: "k" }, "−  /  ="), h("span", null, "mark sample bad / good (also − and + on the numpad)"),
          h("span", { class: "k" }, "1 … 9"), h("span", null, "toggle the tag bound to that number"),
          h("span", { class: "k" }, "Backspace / ←"), h("span", null, "go back to the previous sample (re-deciding overrides)"),
          h("span", { class: "k" }, "→"), h("span", null, "peek at the next sample without deciding"),
          h("span", { class: "k" }, "p / w"), h("span", null, "toggle field view for JSON rows / soft-wrap"),
          h("span", { class: "k" }, "F1 … F7"), h("span", null, "switch screens"),
          h("span", { class: "k" }, "Skip"), h("span", null, "deliberately has no key: it is a button, so parking a sample is always a considered click"),
        ),
      ),
    );
  },
};
