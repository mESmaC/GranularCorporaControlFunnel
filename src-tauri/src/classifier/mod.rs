//! Classifier worker pool. Up to N workers (default cap 8) each own a backend
//! instance — either the built-in heuristic baseline or an external process
//! speaking the JSON-lines protocol (see `classifier/worker.py`). Workers pull
//! batches from a shared queue; predictions are written back to the DB by the
//! coordinating thread.

pub mod builtin;
pub mod command;
pub mod heuristic;
pub mod train;

use crate::db::{self, Db, NewPrediction};
use crate::error::{GccfError, Result};
use crate::jobs::JobHandle;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

pub const MAX_WORKERS: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifyItem {
    pub id: i64,
    pub text: String,
    pub path: String,
    pub format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifyResult {
    pub id: i64,
    pub verdict: String,
    pub score: f64,
    #[serde(default)]
    pub tags: Vec<String>,
}

pub trait Backend: Send {
    fn classify(&mut self, batch: &[ClassifyItem]) -> anyhow::Result<Vec<ClassifyResult>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum BackendSpec {
    Heuristic,
    /// A model trained in-process (`builtin.rs`), loaded once and shared by
    /// every worker thread.
    Builtin {
        path: String,
    },
    Command {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        cwd: Option<String>,
    },
}

impl BackendSpec {
    pub fn name(&self) -> &'static str {
        match self {
            BackendSpec::Heuristic => "heuristic",
            BackendSpec::Builtin { .. } => "builtin",
            BackendSpec::Command { .. } => "command",
        }
    }
    pub fn command_line(&self) -> Option<String> {
        match self {
            BackendSpec::Heuristic => None,
            BackendSpec::Builtin { path } => Some(path.clone()),
            BackendSpec::Command { command, args, .. } => {
                let mut s = command.clone();
                for a in args {
                    s.push(' ');
                    s.push_str(a);
                }
                Some(s)
            }
        }
    }
}

pub fn make_backend(spec: &BackendSpec, worker_index: usize, shared: &Option<Arc<builtin::BuiltinModel>>) -> anyhow::Result<Box<dyn Backend>> {
    match spec {
        BackendSpec::Heuristic => Ok(Box::new(heuristic::Heuristic::default())),
        BackendSpec::Builtin { path } => {
            let model = match shared {
                Some(m) => m.clone(),
                None => Arc::new(builtin::BuiltinModel::load(std::path::Path::new(path))?),
            };
            Ok(Box::new(builtin::BuiltinBackend { model }))
        }
        BackendSpec::Command { command, args, cwd } => Ok(Box::new(command::CommandBackend::spawn(
            command,
            args,
            cwd.as_deref(),
            worker_index,
        )?)),
    }
}

struct BatchResult {
    results: Vec<ClassifyResult>,
    failed_ids: Vec<i64>,
    latency_ms: i64,
    error: Option<String>,
}

pub fn run_pool(
    db: Arc<Db>,
    job: JobHandle,
    run_id: i64,
    spec: BackendSpec,
    workers: usize,
    batch_size: usize,
    items: Vec<ClassifyItem>,
) -> Result<()> {
    let workers = workers.clamp(1, MAX_WORKERS);
    let batch_size = batch_size.max(1);
    let total = items.len() as u64;
    if total == 0 {
        db::finish_run(&db.lock(), run_id, "done", None)?;
        job.finish("done", "nothing to classify");
        return Ok(());
    }

    // Built-in models are loaded once here and shared, not once per worker.
    let shared: Option<Arc<builtin::BuiltinModel>> = match &spec {
        BackendSpec::Builtin { path } => Some(Arc::new(
            builtin::BuiltinModel::load(std::path::Path::new(path)).map_err(|e| GccfError::msg(e.to_string()))?,
        )),
        _ => None,
    };

    let (batch_tx, batch_rx) = crossbeam_channel::unbounded::<Vec<ClassifyItem>>();
    let (res_tx, res_rx) = crossbeam_channel::bounded::<BatchResult>(workers * 2);
    for chunk in items.chunks(batch_size) {
        let _ = batch_tx.send(chunk.to_vec());
    }
    drop(batch_tx);

    let mut handles = Vec::with_capacity(workers);
    for w in 0..workers {
        let rx = batch_rx.clone();
        let tx = res_tx.clone();
        let spec = spec.clone();
        let job = job.clone();
        let shared = shared.clone();
        handles.push(std::thread::Builder::new().name(format!("gccf-clf-{w}")).spawn(move || {
            let mut backend = match make_backend(&spec, w, &shared) {
                Ok(b) => b,
                Err(e) => {
                    // Report every batch we would have taken as failed.
                    for batch in rx.iter() {
                        let _ = tx.send(BatchResult {
                            results: Vec::new(),
                            failed_ids: batch.iter().map(|i| i.id).collect(),
                            latency_ms: 0,
                            error: Some(format!("worker {w} failed to start: {e}")),
                        });
                    }
                    return;
                }
            };
            for batch in rx.iter() {
                if job.is_cancelled() {
                    break;
                }
                let t0 = Instant::now();
                match backend.classify(&batch) {
                    Ok(results) => {
                        let got: std::collections::HashSet<i64> = results.iter().map(|r| r.id).collect();
                        let failed_ids = batch.iter().map(|i| i.id).filter(|id| !got.contains(id)).collect();
                        let _ = tx.send(BatchResult {
                            results,
                            failed_ids,
                            latency_ms: t0.elapsed().as_millis() as i64,
                            error: None,
                        });
                    }
                    Err(e) => {
                        let _ = tx.send(BatchResult {
                            results: Vec::new(),
                            failed_ids: batch.iter().map(|i| i.id).collect(),
                            latency_ms: t0.elapsed().as_millis() as i64,
                            error: Some(e.to_string()),
                        });
                    }
                }
            }
        })?);
    }
    drop(res_tx);

    let mut completed: u64 = 0;
    let mut failed: u64 = 0;
    let mut last_error: Option<String> = None;
    let mut pending: Vec<NewPrediction> = Vec::new();
    let mut last_flush = Instant::now();

    for br in res_rx.iter() {
        for r in br.results {
            let verdict = match r.verdict.to_ascii_lowercase().as_str() {
                "good" | "keep" | "1" | "true" => "good".to_string(),
                "bad" | "drop" | "0" | "false" => "bad".to_string(),
                other => other.to_string(),
            };
            pending.push(NewPrediction {
                sample_id: r.id,
                verdict,
                score: r.score.clamp(0.0, 1.0),
                tags: r.tags.iter().map(|t| db::normalize_tag(t)).filter(|t| !t.is_empty()).collect(),
                latency_ms: Some(br.latency_ms),
            });
            completed += 1;
        }
        failed += br.failed_ids.len() as u64;
        if let Some(e) = br.error {
            log::warn!("classifier batch failed: {e}");
            last_error = Some(e);
        }
        if pending.len() >= 256 || last_flush.elapsed().as_millis() > 500 {
            let mut conn = db.lock();
            db::insert_predictions(&mut conn, run_id, &pending)?;
            db::update_run_progress(&conn, run_id, completed as i64, failed as i64)?;
            pending.clear();
            last_flush = Instant::now();
        }
        job.progress(completed + failed, total, format!("{completed} classified, {failed} failed"));
        if job.is_cancelled() {
            break;
        }
    }
    // Drain: wait for workers to exit (they check the cancel flag per batch).
    for h in handles {
        let _ = h.join();
    }
    {
        let mut conn = db.lock();
        if !pending.is_empty() {
            db::insert_predictions(&mut conn, run_id, &pending)?;
        }
        db::update_run_progress(&conn, run_id, completed as i64, failed as i64)?;
        let status = if job.is_cancelled() {
            "cancelled"
        } else if completed == 0 && failed > 0 {
            "error"
        } else {
            "done"
        };
        db::finish_run(&conn, run_id, status, last_error.as_deref())?;
        let agreement = db::run_agreement(&conn, run_id)?;
        job.finish_with_payload(
            status,
            format!("{completed} classified, {failed} failed"),
            serde_json::json!({ "run_id": run_id, "completed": completed, "failed": failed, "agreement": agreement, "error": last_error }),
        );
    }
    if completed == 0 && failed > 0 {
        return Err(GccfError::msg(last_error.unwrap_or_else(|| "all batches failed".into())));
    }
    Ok(())
}
