import { invoke } from "@tauri-apps/api/core";

// ---------------------------------------------------------------------------
// Types mirrored from src-tauri/src/db.rs & commands.rs
// ---------------------------------------------------------------------------

export interface AppInfo {
  version: string;
  db_path: string;
  cpus: number;
  default_threads: number;
  max_workers: number;
  formats: string[];
}

export interface JobProgress {
  id: number;
  kind: string;
  status: "running" | "done" | "error" | "cancelled";
  done: number;
  total: number;
  message: string;
  started_at: string;
  finished_at: string | null;
  payload: any;
}

export interface StartedJob {
  job_id: number;
  root_id?: number;
  session_id?: number;
  run_id?: number;
}

export interface RootInfo {
  id: number;
  path: string;
  label: string | null;
  created_at: string;
  scanned_at: string | null;
  file_count: number;
  total_bytes: number;
  scan_status: string;
  scan_error: string | null;
  sampleable_files: number;
  sampleable_bytes: number;
  sessions: number;
}

export interface FormatStat {
  format: string;
  kind: string;
  compression: string;
  count: number;
  bytes: number;
  sampled_files: number;
}

export interface StatusCounts {
  pending: number;
  good: number;
  bad: number;
  skip: number;
}

export interface SessionInfo {
  id: number;
  root_id: number;
  root_path: string;
  name: string;
  created_at: string;
  files_requested: number;
  samples_per_file: number;
  lines_per_sample: number;
  seed: number;
  format_filter: string[];
  exclude_reviewed_files: boolean;
  status: string;
  error: string | null;
  files_selected: number;
  samples_total: number;
  counts: StatusCounts;
}

export interface PredictionView {
  run_id: number;
  model_id: string;
  verdict: string;
  score: number;
  tags: string[];
  created_at: string;
}

export interface SampleView {
  id: number;
  session_id: number;
  file_id: number;
  ordinal: number;
  rel_path: string;
  abs_path: string;
  format: string;
  kind: string;
  compression: string;
  file_size: number;
  locator_kind: string;
  locator: any;
  text: string;
  text_bytes: number;
  truncated: boolean;
  content_hash: string;
  status: "pending" | "good" | "bad" | "skip";
  decided_at: string | null;
  decided_by: string | null;
  tags: number[];
  prediction: PredictionView | null;
  duplicates: number;
}

export interface SampleSummary {
  id: number;
  session_id: number;
  session_name: string;
  file_id: number;
  rel_path: string;
  format: string;
  ordinal: number;
  status: string;
  preview: string;
  decided_at: string | null;
  tags: number[];
}

export interface TagInfo {
  id: number;
  name: string;
  builtin: boolean;
  hotkey: number | null;
  usage: number;
}

export interface AgreementStats {
  compared: number;
  agree: number;
  tp: number;
  fp: number;
  tn: number;
  fn_: number;
}

export interface ClassifierRunInfo {
  id: number;
  session_id: number | null;
  session_name: string | null;
  backend: string;
  model_id: string;
  command: string | null;
  workers: number;
  batch_size: number;
  status: string;
  total: number;
  completed: number;
  failed: number;
  started_at: string;
  finished_at: string | null;
  error: string | null;
  agreement: AgreementStats | null;
}

export interface ExportStats {
  rows: number;
  bytes: number;
  good: number;
  bad: number;
  skip: number;
  pending: number;
  human: number;
  classifier: number;
}

export interface ReceiptOptions {
  session_ids: number[];
  verbosity: "summary" | "files" | "full";
  include_min_good_ratio: number;
  exclude_max_good_ratio: number;
  min_decided: number;
  include_unsampled_files: boolean;
  skip_counts_as_bad: boolean;
  include_classifier_decisions: boolean;
  include_text: boolean;
  note: string | null;
}

export interface ReceiptCounts {
  samples: number;
  good: number;
  bad: number;
  skip: number;
  pending: number;
  human: number;
  classifier: number;
}

export interface ReceiptSummary {
  files_include: number;
  files_exclude: number;
  files_review: number;
  files_unsampled: number;
  bytes_include: number;
  bytes_exclude: number;
  bytes_review: number;
  samples: ReceiptCounts;
  exclusions: number;
  tags: Record<string, number>;
}

export interface CreateSessionParams {
  root_id: number;
  name?: string;
  files: number;
  samples_per_file: number;
  lines_per_sample: number;
  seed?: number;
  formats: string[];
  exclude_reviewed_files: boolean;
  max_bytes_per_sample?: number;
  max_stream_bytes?: number;
  threads?: number;
}

export interface RunParams {
  session_id: number | null;
  only_pending: boolean;
  backend: "heuristic" | "command";
  command: string;
  args: string[];
  cwd: string | null;
  model_id: string | null;
  workers: number;
  batch_size: number;
}

export type ReviewMode = "pending" | "skip" | "all" | "decided";

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

export const api = {
  appInfo: () => invoke<AppInfo>("app_info"),
  getSettings: () => invoke<Record<string, string>>("get_settings"),
  setSetting: (key: string, value: string) => invoke<void>("set_setting", { key, value }),
  listJobs: () => invoke<JobProgress[]>("list_jobs"),
  cancelJob: (job_id: number) => invoke<boolean>("cancel_job", { job_id }),

  listRoots: () => invoke<RootInfo[]>("list_roots"),
  addRoot: (path: string, label?: string) => invoke<StartedJob>("add_root", { path, label: label ?? null }),
  rescanRoot: (root_id: number) => invoke<StartedJob>("rescan_root", { root_id }),
  removeRoot: (root_id: number) => invoke<void>("remove_root", { root_id }),
  rootFormatStats: (root_id: number) => invoke<FormatStat[]>("root_format_stats", { root_id }),

  listSessions: () => invoke<SessionInfo[]>("list_sessions"),
  getSession: (session_id: number) => invoke<SessionInfo>("get_session", { session_id }),
  createSession: (params: CreateSessionParams) => invoke<StartedJob>("create_session", { params }),
  deleteSession: (session_id: number) => invoke<void>("delete_session", { session_id }),
  renameSession: (session_id: number, name: string) => invoke<void>("rename_session", { session_id, name }),
  sessionCounts: (session_id: number) => invoke<StatusCounts>("session_counts", { session_id }),

  nextSample: (session_id: number, mode: ReviewMode, after_ordinal: number | null) =>
    invoke<SampleView | null>("next_sample", { session_id, mode, after_ordinal }),
  getSample: (sample_id: number) => invoke<SampleView>("get_sample", { sample_id }),
  decideSample: (sample_id: number, verdict: "good" | "bad" | "skip" | "pending", view_ms: number | null) =>
    invoke<StatusCounts>("decide_sample", { sample_id, verdict, view_ms }),
  setSampleTag: (sample_id: number, tag_id: number, on: boolean) =>
    invoke<number[]>("set_sample_tag", { sample_id, tag_id, on }),
  listSamples: (session_id: number | null, statuses: string[], limit = 200, offset = 0) =>
    invoke<SampleSummary[]>("list_samples", { session_id, statuses, limit, offset }),

  listTags: () => invoke<TagInfo[]>("list_tags"),
  createTag: (name: string) => invoke<number>("create_tag", { name }),
  deleteTag: (tag_id: number) => invoke<void>("delete_tag", { tag_id }),
  renameTag: (tag_id: number, name: string) => invoke<void>("rename_tag", { tag_id, name }),
  setTagHotkey: (tag_id: number, hotkey: number | null) => invoke<void>("set_tag_hotkey", { tag_id, hotkey }),

  revealFile: (file_id: number) => invoke<string>("reveal_file", { file_id }),
  openFile: (file_id: number) => invoke<string>("open_file", { file_id }),
  revealPath: (path: string) => invoke<void>("reveal_path", { path }),

  startClassifierRun: (params: RunParams) => invoke<StartedJob>("start_classifier_run", { params }),
  listClassifierRuns: () => invoke<ClassifierRunInfo[]>("list_classifier_runs"),
  applyRunPredictions: (run_id: number, min_score: number, apply_tags: boolean) =>
    invoke<number>("apply_run_predictions", { run_id, min_score, apply_tags }),
  deleteClassifierRun: (run_id: number) => invoke<void>("delete_classifier_run", { run_id }),

  exportTrainingData: (path: string, session_ids: number[], include_pending: boolean, include_skipped: boolean, human_only: boolean) =>
    invoke<ExportStats>("export_training_data", { path, session_ids, include_pending, include_skipped, human_only }),
  receiptPreview: (options: ReceiptOptions) => invoke<ReceiptSummary>("receipt_preview", { options }),
  exportReceipt: (path: string, options: ReceiptOptions) => invoke<ReceiptSummary>("export_receipt", { path, options }),
};
