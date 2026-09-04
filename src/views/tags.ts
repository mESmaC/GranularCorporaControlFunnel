import { api, type TagInfo } from "../api";
import { h, clear, fmtInt, toast, guard, confirmDialog } from "../ui";
import { refreshTags, type View } from "../state";

let listEl: HTMLElement;

function row(t: TagInfo): HTMLElement {
  const nameIn = h("input", { class: "inline-edit", value: t.name, onChange: async () => { await guard(api.renameTag(t.id, nameIn.value), "rename"); await refresh(); } });
  const hk = h(
    "select",
    { onChange: async () => { await guard(api.setTagHotkey(t.id, hk.value ? Number(hk.value) : null), "hotkey"); await refresh(); } },
    h("option", { value: "" }, "—"),
    ...[1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => h("option", { value: String(n), selected: t.hotkey === n }, String(n))),
  );
  return h(
    "tr",
    null,
    h("td", { style: { width: "40%" } }, nameIn),
    h("td", null, hk),
    h("td", { class: "num" }, fmtInt(t.usage)),
    h("td", null, t.builtin ? h("span", { class: "pill" }, "builtin") : h("span", { class: "pill accent" }, "custom")),
    h("td", null,
      h("button", { class: "sm danger", onClick: async () => {
        if (t.usage > 0 && !confirmDialog(`"${t.name}" is applied to ${t.usage} sample(s). Delete it and remove it from all of them?`)) return;
        await guard(api.deleteTag(t.id), "delete tag");
        await refresh();
      } }, "Delete"),
    ),
  );
}

async function refresh() {
  const tags = await refreshTags();
  clear(listEl);
  listEl.appendChild(
    h("table", null,
      h("thead", null, h("tr", null, h("th", null, "tag"), h("th", null, "hotkey"), h("th", { class: "num" }, "samples"), h("th", null, "origin"), h("th", null, ""))),
      h("tbody", null, ...tags.map(row)),
    ),
  );
}

export const tagsView: View = {
  async render(el) {
    const input = h("input", { type: "text", placeholder: "new tag name", style: { width: "260px" } });
    const add = async () => {
      const name = input.value.trim();
      if (!name) return;
      const id = await guard(api.createTag(name), "create tag");
      if (id !== undefined) {
        input.value = "";
        toast(`Tag added`, "ok", 1500);
        await refresh();
      }
    };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") add(); });
    listEl = h("div");
    el.append(
      h("h1", null, "Tags"),
      h("p", { class: "lead" }, "Tags describe what a sample contains, independent of whether it is good or bad. Nine of them can be bound to number keys for one-press tagging in the reviewer. Names are normalised to lowercase-with-dashes; renaming keeps every existing assignment."),
      h("div", { class: "row", style: { marginBottom: "14px" } }, input, h("button", { class: "primary", onClick: add }, "＋ Add tag")),
      listEl,
    );
    await refresh();
  },
};
