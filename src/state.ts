import { api, type AppInfo, type TagInfo } from "./api";
import { emit } from "./ui";

export interface View {
  render(root: HTMLElement, params?: any): void | Promise<void>;
  onKey?(e: KeyboardEvent): void;
  destroy?(): void;
}

const LS_SESSION = "gccf.sessionId";

export const state = {
  info: null as AppInfo | null,
  tags: [] as TagInfo[],
  sessionId: readSession(),
  settings: {} as Record<string, string>,
};

function readSession(): number | null {
  try {
    const v = localStorage.getItem(LS_SESSION);
    return v ? Number(v) : null;
  } catch {
    return null;
  }
}

export function setSession(id: number | null) {
  state.sessionId = id;
  try {
    if (id === null) localStorage.removeItem(LS_SESSION);
    else localStorage.setItem(LS_SESSION, String(id));
  } catch {
    /* ignore */
  }
  emit("session", id);
}

export async function refreshTags(): Promise<TagInfo[]> {
  state.tags = await api.listTags();
  emit("tags", state.tags);
  return state.tags;
}

export async function refreshSettings(): Promise<Record<string, string>> {
  state.settings = await api.getSettings();
  return state.settings;
}

export function navigate(view: string, params?: any) {
  emit("navigate", { view, params });
}
