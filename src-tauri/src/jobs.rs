//! Background job registry. Long-running work (scans, session builds,
//! classifier runs) runs on its own thread and reports progress here; the
//! frontend receives throttled `job:progress` events and can cancel.

use serde::Serialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

pub const EVENT_NAME: &str = "job:progress";

#[derive(Debug, Clone, Serialize)]
pub struct JobProgress {
    pub id: u64,
    pub kind: String,
    pub status: String, // running | done | error | cancelled
    pub done: u64,
    pub total: u64,
    pub message: String,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub payload: serde_json::Value,
}

struct Entry {
    progress: JobProgress,
    cancel: Arc<AtomicBool>,
}

pub struct Jobs {
    next: AtomicU64,
    inner: Mutex<HashMap<u64, Entry>>,
}

impl Default for Jobs {
    fn default() -> Self {
        Self::new()
    }
}

impl Jobs {
    pub fn new() -> Self {
        Jobs {
            next: AtomicU64::new(1),
            inner: Mutex::new(HashMap::new()),
        }
    }

    pub fn start(self: &Arc<Self>, app: Option<AppHandle>, kind: &str, total: u64, message: &str, payload: serde_json::Value) -> JobHandle {
        let id = self.next.fetch_add(1, Ordering::SeqCst);
        let cancel = Arc::new(AtomicBool::new(false));
        let progress = JobProgress {
            id,
            kind: kind.to_string(),
            status: "running".into(),
            done: 0,
            total,
            message: message.to_string(),
            started_at: crate::db::now_iso(),
            finished_at: None,
            payload,
        };
        self.inner.lock().unwrap().insert(
            id,
            Entry {
                progress: progress.clone(),
                cancel: cancel.clone(),
            },
        );
        let handle = JobHandle {
            id,
            jobs: self.clone(),
            app,
            cancel,
            last_emit: Arc::new(Mutex::new(Instant::now() - Duration::from_secs(1))),
        };
        handle.emit(&progress);
        handle
    }

    pub fn list(&self) -> Vec<JobProgress> {
        let mut v: Vec<JobProgress> = self.inner.lock().unwrap().values().map(|e| e.progress.clone()).collect();
        v.sort_by_key(|p| p.id);
        v
    }

    pub fn get(&self, id: u64) -> Option<JobProgress> {
        self.inner.lock().unwrap().get(&id).map(|e| e.progress.clone())
    }

    pub fn cancel(&self, id: u64) -> bool {
        if let Some(e) = self.inner.lock().unwrap().get(&id) {
            e.cancel.store(true, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    /// Drop finished jobs older than `keep` entries to bound memory.
    pub fn prune(&self, keep: usize) {
        let mut g = self.inner.lock().unwrap();
        let mut finished: Vec<u64> = g
            .iter()
            .filter(|(_, e)| e.progress.status != "running")
            .map(|(id, _)| *id)
            .collect();
        finished.sort();
        if finished.len() > keep {
            for id in &finished[..finished.len() - keep] {
                g.remove(id);
            }
        }
    }
}

#[derive(Clone)]
pub struct JobHandle {
    pub id: u64,
    jobs: Arc<Jobs>,
    app: Option<AppHandle>,
    cancel: Arc<AtomicBool>,
    last_emit: Arc<Mutex<Instant>>,
}

impl JobHandle {
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }

    fn emit(&self, p: &JobProgress) {
        if let Some(app) = &self.app {
            let _ = app.emit(EVENT_NAME, p);
        }
    }

    fn update<F: FnOnce(&mut JobProgress)>(&self, force: bool, f: F) {
        let snapshot = {
            let mut g = self.jobs.inner.lock().unwrap();
            match g.get_mut(&self.id) {
                Some(e) => {
                    f(&mut e.progress);
                    e.progress.clone()
                }
                None => return,
            }
        };
        let should_emit = force || {
            let mut last = self.last_emit.lock().unwrap();
            if last.elapsed() >= Duration::from_millis(120) {
                *last = Instant::now();
                true
            } else {
                false
            }
        };
        if should_emit {
            self.emit(&snapshot);
        }
    }

    pub fn progress(&self, done: u64, total: u64, message: impl Into<String>) {
        let message = message.into();
        self.update(false, |p| {
            p.done = done;
            p.total = total;
            p.message = message;
        });
    }

    pub fn set_payload(&self, payload: serde_json::Value) {
        self.update(false, |p| p.payload = payload);
    }

    pub fn finish(&self, status: &str, message: impl Into<String>) {
        let message = message.into();
        let status = status.to_string();
        self.update(true, |p| {
            p.status = status;
            p.message = message;
            p.finished_at = Some(crate::db::now_iso());
        });
        self.jobs.prune(50);
    }

    pub fn finish_with_payload(&self, status: &str, message: impl Into<String>, payload: serde_json::Value) {
        let message = message.into();
        let status = status.to_string();
        self.update(true, |p| {
            p.status = status;
            p.message = message;
            p.payload = payload;
            p.finished_at = Some(crate::db::now_iso());
        });
        self.jobs.prune(50);
    }
}
