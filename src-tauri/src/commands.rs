//! Tauri command surface. Every command is `async` so it runs off the main
//! thread; long work is handed to a dedicated thread and tracked as a job.

use crate::classifier::{self, BackendSpec, ClassifyItem};
use crate::db::{self, Db};
use crate::error::{GccfError, Result};
use crate::export::{self, ExportStats};
use crate::jobs::{JobProgress, Jobs};
use crate::receipt::{self, ReceiptOptions, ReceiptSummary};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tauri::{AppHandle, State};
use tauri_plugin_opener::OpenerExt;

pub struct AppState {
    pub db: Arc<Db>,
    pub jobs: Arc<Jobs>,
}

#[derive(Serialize)]
pub struct AppInfo {
    pub version: String,
    pub db_path: String,
    pub cpus: usize,
    pub default_threads: usize,
    pub max_workers: usize,
    pub formats: Vec<String>,
}

#[derive(Serialize)]
pub struct StartedJob {
    pub job_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<i64>,
}

fn setting_usize(db: &Db, key: &str, default: usize) -> usize {
    db::get_setting(&db.lock(), key)
        .ok()
        .flatten()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn default_threads() -> usize {
    num_cpus::get().clamp(1, 8)
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

#[tauri::command(rename_all = "snake_case")]
pub async fn app_info(state: State<'_, AppState>) -> Result<AppInfo> {
    Ok(AppInfo {
        version: env!("CARGO_PKG_VERSION").into(),
        db_path: state.db.path.to_string_lossy().to_string(),
        cpus: num_cpus::get(),
        default_threads: setting_usize(&state.db, "sample_threads", default_threads()),
        max_workers: classifier::MAX_WORKERS,
        formats: crate::formats::Format::all().iter().map(|f| f.as_str().to_string()).collect(),
    })
}

#[tauri::command(rename_all = "snake_case")]
pub async fn get_settings(state: State<'_, AppState>) -> Result<HashMap<String, String>> {
    db::all_settings(&state.db.lock())
}

#[tauri::command(rename_all = "snake_case")]
pub async fn set_setting(state: State<'_, AppState>, key: String, value: String) -> Result<()> {
    db::set_setting(&state.db.lock(), &key, &value)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn list_jobs(state: State<'_, AppState>) -> Result<Vec<JobProgress>> {
    Ok(state.jobs.list())
}

#[tauri::command(rename_all = "snake_case")]
pub async fn cancel_job(state: State<'_, AppState>, job_id: u64) -> Result<bool> {
    Ok(state.jobs.cancel(job_id))
}

// ---------------------------------------------------------------------------
// Roots
// ---------------------------------------------------------------------------

#[tauri::command(rename_all = "snake_case")]
pub async fn list_roots(state: State<'_, AppState>) -> Result<Vec<db::RootInfo>> {
    db::list_roots(&state.db.lock())
}

fn spawn_scan(app: AppHandle, state: &AppState, root_id: i64, path: PathBuf) -> u64 {
    let job = state
        .jobs
        .start(Some(app), "scan", 0, &format!("scanning {}", path.display()), serde_json::json!({ "root_id": root_id }));
    let job_id = job.id;
    let db = state.db.clone();
    std::thread::Builder::new()
        .name("gccf-scan".into())
        .spawn(move || match crate::scan::scan_root(&db, &job, root_id, &path) {
            Ok((count, bytes)) => job.finish_with_payload(
                if job.is_cancelled() { "cancelled" } else { "done" },
                format!("{count} files, {:.2} GB", bytes as f64 / 1e9),
                serde_json::json!({ "root_id": root_id, "files": count, "bytes": bytes }),
            ),
            Err(e) => {
                let _ = db::set_root_scan_status(&db.lock(), root_id, "error", Some(&e.to_string()));
                job.finish("error", e.to_string());
            }
        })
        .expect("spawn scan thread");
    job_id
}

#[tauri::command(rename_all = "snake_case")]
pub async fn add_root(app: AppHandle, state: State<'_, AppState>, path: String, label: Option<String>) -> Result<StartedJob> {
    let p = PathBuf::from(&path);
    if !p.is_dir() {
        return Err(GccfError::msg(format!("not a directory: {path}")));
    }
    let canonical = dunce_canonicalize(&p);
    let root_id = {
        let conn = state.db.lock();
        match conn
            .query_row("SELECT id FROM roots WHERE path = ?1", [canonical.to_string_lossy().to_string()], |r| r.get::<_, i64>(0))
        {
            Ok(id) => id,
            Err(_) => db::insert_root(&conn, &canonical.to_string_lossy(), label.as_deref())?,
        }
    };
    let job_id = spawn_scan(app, &state, root_id, canonical);
    Ok(StartedJob {
        job_id,
        root_id: Some(root_id),
        session_id: None,
        run_id: None,
    })
}

/// `std::fs::canonicalize` yields `\\?\C:\...` on Windows, which is ugly in a
/// UI and confuses some tools; strip the prefix when it is safe to do so.
fn dunce_canonicalize(p: &Path) -> PathBuf {
    match std::fs::canonicalize(p) {
        Ok(c) => {
            let s = c.to_string_lossy().to_string();
            if let Some(stripped) = s.strip_prefix(r"\\?\") {
                if !stripped.starts_with("UNC\\") {
                    return PathBuf::from(stripped);
                }
            }
            c
        }
        Err(_) => p.to_path_buf(),
    }
}

#[tauri::command(rename_all = "snake_case")]
pub async fn rescan_root(app: AppHandle, state: State<'_, AppState>, root_id: i64) -> Result<StartedJob> {
    let path = db::get_root_path(&state.db.lock(), root_id)?;
    let job_id = spawn_scan(app, &state, root_id, PathBuf::from(path));
    Ok(StartedJob {
        job_id,
        root_id: Some(root_id),
        session_id: None,
        run_id: None,
    })
}

#[tauri::command(rename_all = "snake_case")]
pub async fn remove_root(state: State<'_, AppState>, root_id: i64) -> Result<()> {
    db::delete_root(&state.db.lock(), root_id)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn root_format_stats(state: State<'_, AppState>, root_id: i64) -> Result<Vec<db::FormatStat>> {
    db::root_format_stats(&state.db.lock(), root_id)
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct CreateSessionParams {
    pub root_id: i64,
    pub name: Option<String>,
    pub files: i64,
    pub samples_per_file: i64,
    pub lines_per_sample: i64,
    pub seed: Option<i64>,
    #[serde(default)]
    pub formats: Vec<String>,
    #[serde(default = "default_true")]
    pub exclude_reviewed_files: bool,
    pub max_bytes_per_sample: Option<usize>,
    pub max_stream_bytes: Option<u64>,
    pub threads: Option<usize>,
}

fn default_true() -> bool {
    true
}

#[tauri::command(rename_all = "snake_case")]
pub async fn list_sessions(state: State<'_, AppState>) -> Result<Vec<db::SessionInfo>> {
    db::list_sessions(&state.db.lock())
}

#[tauri::command(rename_all = "snake_case")]
pub async fn get_session(state: State<'_, AppState>, session_id: i64) -> Result<db::SessionInfo> {
    db::get_session(&state.db.lock(), session_id)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn create_session(app: AppHandle, state: State<'_, AppState>, params: CreateSessionParams) -> Result<StartedJob> {
    if params.files <= 0 || params.samples_per_file <= 0 || params.lines_per_sample <= 0 {
        return Err(GccfError::msg("files, samples per file and lines per sample must be positive"));
    }
    let seed = params.seed.unwrap_or_else(|| (chrono::Utc::now().timestamp_millis() & 0x7fff_ffff) as i64);
    let name = params
        .name
        .filter(|n| !n.trim().is_empty())
        .unwrap_or_else(|| format!("session {}", chrono::Local::now().format("%Y-%m-%d %H:%M")));
    let session_id = {
        let conn = state.db.lock();
        db::insert_session(
            &conn,
            &db::NewSession {
                root_id: params.root_id,
                name: &name,
                files_requested: params.files,
                samples_per_file: params.samples_per_file,
                lines_per_sample: params.lines_per_sample,
                seed,
                format_filter: &params.formats,
                exclude_reviewed_files: params.exclude_reviewed_files,
            },
        )?
    };
    let cfg = crate::session::BuildConfig {
        threads: params.threads.unwrap_or_else(|| setting_usize(&state.db, "sample_threads", default_threads())),
        max_bytes_per_sample: params
            .max_bytes_per_sample
            .unwrap_or_else(|| setting_usize(&state.db, "max_bytes_per_sample", 256 * 1024)),
        max_stream_bytes: params
            .max_stream_bytes
            .unwrap_or_else(|| setting_usize(&state.db, "max_stream_bytes", 0) as u64),
    };
    let job = state.jobs.start(
        Some(app),
        "session",
        0,
        &format!("building {name}"),
        serde_json::json!({ "session_id": session_id }),
    );
    let job_id = job.id;
    let db = state.db.clone();
    std::thread::Builder::new()
        .name("gccf-session".into())
        .spawn(move || {
            if let Err(e) = crate::session::build_session(db.clone(), job.clone(), session_id, cfg) {
                let _ = db::finish_session(&db.lock(), session_id, 0, "error", Some(&e.to_string()));
                job.finish("error", e.to_string());
            }
        })
        .expect("spawn session thread");
    Ok(StartedJob {
        job_id,
        root_id: None,
        session_id: Some(session_id),
        run_id: None,
    })
}

#[tauri::command(rename_all = "snake_case")]
pub async fn delete_session(state: State<'_, AppState>, session_id: i64) -> Result<()> {
    db::delete_session(&state.db.lock(), session_id)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn rename_session(state: State<'_, AppState>, session_id: i64, name: String) -> Result<()> {
    db::rename_session(&state.db.lock(), session_id, name.trim())
}

// ---------------------------------------------------------------------------
// Review
// ---------------------------------------------------------------------------

fn statuses_for_mode(mode: &str) -> Vec<&'static str> {
    match mode {
        "skip" | "queue" => vec!["skip"],
        "all" => vec!["pending", "good", "bad", "skip"],
        "decided" => vec!["good", "bad", "skip"],
        _ => vec!["pending"],
    }
}

#[tauri::command(rename_all = "snake_case")]
pub async fn next_sample(state: State<'_, AppState>, session_id: i64, mode: String, after_ordinal: Option<i64>) -> Result<Option<db::SampleView>> {
    db::next_sample(&state.db.lock(), session_id, &statuses_for_mode(&mode), after_ordinal)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn get_sample(state: State<'_, AppState>, sample_id: i64) -> Result<db::SampleView> {
    db::get_sample(&state.db.lock(), sample_id)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn decide_sample(state: State<'_, AppState>, sample_id: i64, verdict: String, view_ms: Option<i64>) -> Result<db::StatusCounts> {
    let conn = state.db.lock();
    db::decide(&conn, sample_id, &verdict, "human", None, None, view_ms)?;
    let session_id: i64 = conn.query_row("SELECT session_id FROM samples WHERE id = ?1", [sample_id], |r| r.get(0))?;
    db::session_counts(&conn, session_id)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn set_sample_tag(state: State<'_, AppState>, sample_id: i64, tag_id: i64, on: bool) -> Result<Vec<i64>> {
    let conn = state.db.lock();
    db::set_sample_tag(&conn, sample_id, tag_id, on, "human")?;
    Ok(db::sample_tag_ids(&conn, sample_id)?)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn list_samples(
    state: State<'_, AppState>,
    session_id: Option<i64>,
    statuses: Vec<String>,
    limit: Option<i64>,
    offset: Option<i64>,
) -> Result<Vec<db::SampleSummary>> {
    let st: Vec<&str> = statuses.iter().map(|s| s.as_str()).collect();
    db::list_samples(&state.db.lock(), session_id, &st, limit.unwrap_or(200), offset.unwrap_or(0))
}

#[tauri::command(rename_all = "snake_case")]
pub async fn session_counts(state: State<'_, AppState>, session_id: i64) -> Result<db::StatusCounts> {
    db::session_counts(&state.db.lock(), session_id)
}

// ---------------------------------------------------------------------------
// Tags
// ---------------------------------------------------------------------------

#[tauri::command(rename_all = "snake_case")]
pub async fn list_tags(state: State<'_, AppState>) -> Result<Vec<db::TagInfo>> {
    db::list_tags(&state.db.lock())
}

#[tauri::command(rename_all = "snake_case")]
pub async fn create_tag(state: State<'_, AppState>, name: String) -> Result<i64> {
    db::create_tag(&state.db.lock(), &name)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn delete_tag(state: State<'_, AppState>, tag_id: i64) -> Result<()> {
    db::delete_tag(&state.db.lock(), tag_id)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn rename_tag(state: State<'_, AppState>, tag_id: i64, name: String) -> Result<()> {
    db::rename_tag(&state.db.lock(), tag_id, &name)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn set_tag_hotkey(state: State<'_, AppState>, tag_id: i64, hotkey: Option<i64>) -> Result<()> {
    db::set_tag_hotkey(&state.db.lock(), tag_id, hotkey)
}

// ---------------------------------------------------------------------------
// External editing
// ---------------------------------------------------------------------------

#[tauri::command(rename_all = "snake_case")]
pub async fn reveal_file(app: AppHandle, state: State<'_, AppState>, file_id: i64) -> Result<String> {
    let f = db::get_file(&state.db.lock(), file_id)?;
    let p = f.abs_path();
    app.opener()
        .reveal_item_in_dir(&p)
        .map_err(|e| GccfError::msg(format!("reveal failed: {e}")))?;
    Ok(p.to_string_lossy().to_string())
}

#[tauri::command(rename_all = "snake_case")]
pub async fn open_file(app: AppHandle, state: State<'_, AppState>, file_id: i64) -> Result<String> {
    let f = db::get_file(&state.db.lock(), file_id)?;
    let p = f.abs_path();
    app.opener()
        .open_path(p.to_string_lossy().to_string(), None::<&str>)
        .map_err(|e| GccfError::msg(format!("open failed: {e}")))?;
    Ok(p.to_string_lossy().to_string())
}

#[tauri::command(rename_all = "snake_case")]
pub async fn reveal_path(app: AppHandle, path: String) -> Result<()> {
    app.opener()
        .reveal_item_in_dir(&path)
        .map_err(|e| GccfError::msg(format!("reveal failed: {e}")))
}

// ---------------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct RunParams {
    pub session_id: Option<i64>,
    #[serde(default = "default_true")]
    pub only_pending: bool,
    /// "heuristic" | "command"
    pub backend: String,
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    pub cwd: Option<String>,
    pub model_id: Option<String>,
    pub workers: Option<usize>,
    pub batch_size: Option<usize>,
}

#[tauri::command(rename_all = "snake_case")]
pub async fn start_classifier_run(app: AppHandle, state: State<'_, AppState>, params: RunParams) -> Result<StartedJob> {
    let spec = match params.backend.as_str() {
        "heuristic" => BackendSpec::Heuristic,
        "command" => {
            if params.command.trim().is_empty() {
                return Err(GccfError::msg("classifier command is empty"));
            }
            BackendSpec::Command {
                command: params.command.clone(),
                args: params.args.clone(),
                cwd: params.cwd.clone(),
            }
        }
        other => return Err(GccfError::msg(format!("unknown backend {other}"))),
    };
    let model_id = params
        .model_id
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| match &spec {
            BackendSpec::Heuristic => "heuristic-v1".into(),
            BackendSpec::Command { command, .. } => {
                // last token that looks like a model path, else the command itself
                command
                    .split_whitespace()
                    .filter(|t| t.contains('.') && !t.ends_with(".py"))
                    .last()
                    .unwrap_or(command)
                    .to_string()
            }
        });
    let workers = params.workers.unwrap_or(4).clamp(1, classifier::MAX_WORKERS);
    let batch_size = params.batch_size.unwrap_or(16).max(1);

    let items: Vec<ClassifyItem> = {
        let conn = state.db.lock();
        db::samples_for_classification(&conn, params.session_id, params.only_pending)?
            .into_iter()
            .map(|(id, text, path, format)| ClassifyItem { id, text, path, format })
            .collect()
    };
    let run_id = db::insert_run(
        &state.db.lock(),
        params.session_id,
        spec.name(),
        &model_id,
        spec.command_line().as_deref(),
        workers as i64,
        batch_size as i64,
        items.len() as i64,
    )?;
    let job = state.jobs.start(
        Some(app),
        "classify",
        items.len() as u64,
        &format!("classifying {} samples with {model_id}", items.len()),
        serde_json::json!({ "run_id": run_id, "session_id": params.session_id }),
    );
    let job_id = job.id;
    let db = state.db.clone();
    std::thread::Builder::new()
        .name("gccf-classify".into())
        .spawn(move || {
            if let Err(e) = classifier::run_pool(db.clone(), job.clone(), run_id, spec, workers, batch_size, items) {
                let _ = db::finish_run(&db.lock(), run_id, "error", Some(&e.to_string()));
                job.finish("error", e.to_string());
            }
        })
        .expect("spawn classifier thread");
    Ok(StartedJob {
        job_id,
        root_id: None,
        session_id: params.session_id,
        run_id: Some(run_id),
    })
}

#[tauri::command(rename_all = "snake_case")]
pub async fn list_classifier_runs(state: State<'_, AppState>) -> Result<Vec<db::ClassifierRunInfo>> {
    db::list_runs(&state.db.lock())
}

#[tauri::command(rename_all = "snake_case")]
pub async fn apply_run_predictions(state: State<'_, AppState>, run_id: i64, min_score: f64, apply_tags: bool) -> Result<i64> {
    db::apply_run_predictions(&mut state.db.lock(), run_id, min_score, apply_tags)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn delete_classifier_run(state: State<'_, AppState>, run_id: i64) -> Result<()> {
    state
        .db
        .lock()
        .execute("DELETE FROM classifier_runs WHERE id = ?1 AND status != 'running'", [run_id])?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

#[tauri::command(rename_all = "snake_case")]
pub async fn export_training_data(
    state: State<'_, AppState>,
    path: String,
    session_ids: Vec<i64>,
    include_pending: bool,
    include_skipped: bool,
    human_only: bool,
) -> Result<ExportStats> {
    export::export_training_jsonl(
        &state.db.lock(),
        Path::new(&path),
        &session_ids,
        include_pending,
        include_skipped,
        human_only,
    )
}

#[tauri::command(rename_all = "snake_case")]
pub async fn receipt_preview(state: State<'_, AppState>, options: ReceiptOptions) -> Result<ReceiptSummary> {
    let mut opts = options;
    opts.verbosity = "summary".into();
    Ok(receipt::build_receipt(&state.db.lock(), &opts)?.summary)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn export_receipt(state: State<'_, AppState>, path: String, options: ReceiptOptions) -> Result<ReceiptSummary> {
    receipt::write_receipt(&state.db.lock(), Path::new(&path), &options)
}
