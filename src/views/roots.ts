import { open } from "@tauri-apps/plugin-dialog";
import { api, type RootInfo, type FormatStat } from "../api";
import { h, clear, fmtBytes, fmtInt, fmtDate, toast, guard, on, confirmDialog } from "../ui";
import { navigate, type View } from "../state";

let root: HTMLElement;
let listEl: HTMLElement;
let unsub: (() => void) | null = null;
const expanded = new Set<number>();

async function addFolder() {
  const dir = await open({ directory: true, multiple: false, title: "Select a corpus folder to index" });
  if (!dir || Array.isArray(dir)) return;
  const res = await guard(api.addRoot(dir), "add folder");
  if (res) toast(`Indexing ${dir}…`, "info");
  await refresh();
}

function pill(status: string) {
  const cls = status === "done" ? "good" : status === "error" ? "bad" : status === "scanning" ? "accent" : "pending";
  return h("span", { class: `pill ${cls}` }, status);
}

async function formatTable(r: RootInfo): Promise<HTMLElement> {
  const stats = (await guard(api.rootFormatStats(r.id), "format stats")) ?? [];
  if (!stats.length) return h("div", { class: "muted small" }, "No files indexed yet.");
  const row = (s: FormatStat) =>
    h(
      "tr",
      null,
      h("td", null, h("span", { class: "pill" }, s.format)),
      h("td", null, s.kind === "unsupported" ? h("span", { class: "pill bad" }, "unsupported") : h("span", { class: "pill accent" }, s.kind)),
      h("td", null, s.compression === "none" ? h("span", { class: "muted" }, "–") : s.compression),
      h("td", { class: "num" }, fmtInt(s.count)),
      h("td", { class: "num" }, fmtBytes(s.bytes)),
      h("td", { class: "num" }, fmtInt(s.sampled_files)),
    );
  return h(
    "table",
    { style: { marginTop: "8px" } },
    h("thead", null, h("tr", null, h("th", null, "format"), h("th", null, "sampling"), h("th", null, "compression"), h("th", { class: "num" }, "files"), h("th", { class: "num" }, "bytes"), h("th", { class: "num" }, "files sampled"))),
    h("tbody", null, ...stats.map(row)),
  );
}

async function refresh() {
  const roots = (await guard(api.listRoots(), "list corpora")) ?? [];
  clear(listEl);
  if (!roots.length) {
    listEl.appendChild(h("div", { class: "empty" }, "No corpora indexed yet. Add a folder to build its file index — nothing is copied, only catalogued."));
    return;
  }
  for (const r of roots) {
    const details = h("div", { class: "root-details" });
    const isOpen = expanded.has(r.id);
    const card = h(
      "div",
      { class: "card" },
      h(
        "div",
        { class: "row between" },
        h("div", { class: "stack", style: { gap: "2px" } }, h("div", { class: "path", style: { fontSize: "13px" } }, r.path), h("div", { class: "row small muted" }, pill(r.scan_status), r.scan_error ? h("span", { class: "muted" }, r.scan_error) : null, h("span", null, `scanned ${fmtDate(r.scanned_at)}`))),
        h(
          "div",
          { class: "row" },
          h("button", { class: "sm", onClick: () => guard(api.rescanRoot(r.id), "rescan").then(refresh) }, "Rescan"),
          h(
            "button",
            { class: "sm", onClick: async () => { if (expanded.has(r.id)) expanded.delete(r.id); else expanded.add(r.id); await refresh(); } },
            isOpen ? "Hide formats" : "Formats",
          ),
          h("button", { class: "sm primary", onClick: () => navigate("sessions", { rootId: r.id }) }, "New session"),
          h(
            "button",
            { class: "sm danger", onClick: async () => {
              const msg = r.sessions > 0
                ? `Remove ${r.path}?\n\nThis deletes ${r.sessions} session(s) and all annotations made on files from this corpus. The files on disk are not touched.`
                : `Remove ${r.path} from the index? Files on disk are not touched.`;
              if (!confirmDialog(msg)) return;
              await guard(api.removeRoot(r.id), "remove");
              await refresh();
            } },
            "Remove",
          ),
        ),
      ),
      h(
        "div",
        { class: "stat-tiles", style: { marginTop: "12px" } },
        tile(fmtInt(r.file_count), "files"),
        tile(fmtBytes(r.total_bytes), "total size"),
        tile(fmtInt(r.sampleable_files), "sampleable files"),
        tile(fmtBytes(r.sampleable_bytes), "sampleable bytes"),
        tile(fmtInt(r.sessions), "sessions"),
      ),
      details,
    );
    listEl.appendChild(card);
    if (isOpen) details.appendChild(await formatTable(r));
  }
}

function tile(v: string, l: string, cls = "") {
  return h("div", { class: `tile ${cls}` }, h("div", { class: "v" }, v), h("div", { class: "l" }, l));
}

export const rootsView: View = {
  async render(el) {
    root = el;
    listEl = h("div", { class: "stack" });
    root.append(
      h("h1", null, "Corpora"),
      h("p", { class: "lead" }, "Each corpus folder is walked once to build a file index (path, size, format, compression). Files are only opened later, when a session samples them. Re-scan after editing files so changed sizes are picked up."),
      h("div", { class: "row", style: { marginBottom: "14px" } }, h("button", { class: "primary", onClick: addFolder }, "＋ Add corpus folder")),
      listEl,
    );
    await refresh();
    unsub = on("job", (j) => {
      if (j.kind === "scan" && j.status !== "running") refresh();
      else if (j.kind === "scan" && j.done % 20000 === 0) refresh();
    });
  },
  destroy() {
    unsub?.();
    unsub = null;
  },
};
