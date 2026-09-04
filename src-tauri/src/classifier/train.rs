//! Training. Two trainers share the same inputs (human decisions + tags +
//! evaluation feedback from the catalogue) and the same output (a row in the
//! `models` table):
//!
//! * `train_builtin` — in-process logistic regression over hashed n-grams
//!   (see `builtin.rs`). Supports finetuning: warm-start from a parent model
//!   and up-weight samples the parent got wrong during evaluation.
//! * `train_external` — exports the training JSONL, runs a user command with
//!   `{data}` / `{model}` / `{parent_model}` placeholders, streams its output
//!   into the job log, and registers the resulting artifact with the worker
//!   command that serves it.

use super::builtin::{self, BuiltinModel, FeatureConfig, ModelHeader, Sparse, DENSE};
use super::command::split_command_line;
use crate::db::{self, Db};
use crate::error::{GccfError, Result};
use crate::jobs::JobHandle;
use crate::rng::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn d_true() -> bool {
    true
}
fn d_holdout() -> f64 {
    0.2
}
fn d_min_tag() -> usize {
    5
}
fn d_epochs() -> usize {
    20
}
fn d_feedback_weight() -> f64 {
    3.0
}
fn d_seed() -> u64 {
    42
}
fn d_ext() -> String {
    "joblib".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrainOptions {
    pub name: String,
    #[serde(default)]
    pub session_ids: Vec<i64>,
    #[serde(default = "d_true")]
    pub human_only: bool,
    #[serde(default = "d_holdout")]
    pub holdout: f64,
    #[serde(default = "d_min_tag")]
    pub min_tag_count: usize,
    #[serde(default = "d_epochs")]
    pub epochs: usize,
    /// Finetune: warm-start from this model (must be a built-in model).
    #[serde(default)]
    pub parent_model: Option<i64>,
    /// Weight multiplier for samples a model got wrong during evaluation.
    #[serde(default = "d_feedback_weight")]
    pub feedback_weight: f64,
    #[serde(default = "d_seed")]
    pub seed: u64,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct HeadMetrics {
    pub n_train: usize,
    pub n_holdout: usize,
    pub positives: usize,
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub fn_: usize,
    pub accuracy: Option<f64>,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TagMetrics {
    pub name: String,
    pub count: usize,
    pub metrics: HeadMetrics,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct TrainMetrics {
    pub rows: usize,
    pub labelled: usize,
    pub good: usize,
    pub bad: usize,
    pub feedback_rows: usize,
    pub warm_start: bool,
    pub epochs: usize,
    pub verdict: HeadMetrics,
    pub tags: Vec<TagMetrics>,
    pub elapsed_ms: u128,
}

struct TrainRow {
    label: Option<u8>,
    tags: Vec<String>,
    weight: f32,
    x: Sparse,
}

pub fn slug(name: &str) -> String {
    let s: String = name
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    let s = s.trim_matches('-').to_string();
    if s.is_empty() {
        "model".into()
    } else {
        s.chars().take(48).collect()
    }
}

fn stamp() -> String {
    chrono::Local::now().format("%Y%m%d-%H%M%S").to_string()
}

fn head_metrics(preds: &[(bool, bool)]) -> HeadMetrics {
    let mut m = HeadMetrics {
        n_holdout: preds.len(),
        ..Default::default()
    };
    for &(pred, truth) in preds {
        match (pred, truth) {
            (true, true) => m.tp += 1,
            (true, false) => m.fp += 1,
            (false, false) => m.tn += 1,
            (false, true) => m.fn_ += 1,
        }
    }
    if !preds.is_empty() {
        m.accuracy = Some((m.tp + m.tn) as f64 / preds.len() as f64);
    }
    if m.tp + m.fp > 0 {
        m.precision = Some(m.tp as f64 / (m.tp + m.fp) as f64);
    }
    if m.tp + m.fn_ > 0 {
        m.recall = Some(m.tp as f64 / (m.tp + m.fn_) as f64);
    }
    if let (Some(p), Some(r)) = (m.precision, m.recall) {
        if p + r > 0.0 {
            m.f1 = Some(2.0 * p * r / (p + r));
        }
    }
    m
}

/// Plain SGD logistic regression on sparse rows. `targets[i]` is 0/1,
/// `sample_weight[i]` scales the gradient (class balance, feedback weight).
#[allow(clippy::too_many_arguments)]
fn train_head(
    rows: &[TrainRow],
    idx: &[usize],
    targets: &[f32],
    sample_weight: &[f32],
    init: Option<&[f32]>,
    n_features: usize,
    epochs: usize,
    lr0: f32,
    rng: &mut Rng,
    mut on_epoch: impl FnMut(usize),
) -> Vec<f32> {
    let mut w: Vec<f32> = match init {
        Some(v) if v.len() == n_features + 1 => v.to_vec(),
        _ => vec![0.0; n_features + 1],
    };
    let l2: f32 = 1e-6;
    let mut order: Vec<usize> = (0..idx.len()).collect();
    for epoch in 0..epochs {
        rng.shuffle(&mut order);
        let lr = lr0 / (1.0 + epoch as f32 * 0.3);
        for &k in &order {
            let row = &rows[idx[k]];
            let p = builtin::sigmoid(builtin::dot(&w, &row.x));
            let g = (p - targets[k]) * sample_weight[k];
            if g == 0.0 {
                continue;
            }
            for &(j, v) in &row.x {
                let wj = &mut w[j as usize];
                *wj -= lr * (g * v + l2 * *wj);
            }
            w[n_features] -= lr * g;
        }
        on_epoch(epoch + 1);
    }
    w
}

/// Balanced sample weights: each class contributes equally overall.
fn balanced_weights(rows: &[TrainRow], idx: &[usize], targets: &[f32]) -> Vec<f32> {
    let pos = targets.iter().filter(|&&t| t > 0.5).count().max(1) as f32;
    let neg = (targets.len() - pos as usize).max(1) as f32;
    let n = targets.len() as f32;
    idx.iter()
        .zip(targets)
        .map(|(&i, &t)| rows[i].weight * if t > 0.5 { n / (2.0 * pos) } else { n / (2.0 * neg) })
        .collect()
}

fn split_stratified(rows: &[TrainRow], idx: &[usize], targets: &[f32], holdout: f64, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
    // Returns (train positions, holdout positions) into `idx`/`targets`.
    let mut pos: Vec<usize> = Vec::new();
    let mut neg: Vec<usize> = Vec::new();
    for (k, &t) in targets.iter().enumerate() {
        let _ = &rows[idx[k]];
        if t > 0.5 {
            pos.push(k);
        } else {
            neg.push(k);
        }
    }
    rng.shuffle(&mut pos);
    rng.shuffle(&mut neg);
    let take = |v: &Vec<usize>| -> usize {
        if v.len() < 5 {
            0
        } else {
            ((v.len() as f64 * holdout).round() as usize).clamp(1, v.len() - 2)
        }
    };
    let hp = take(&pos);
    let hn = take(&neg);
    let mut hold: Vec<usize> = pos[..hp].to_vec();
    hold.extend_from_slice(&neg[..hn]);
    let mut train: Vec<usize> = pos[hp..].to_vec();
    train.extend_from_slice(&neg[hn..]);
    (train, hold)
}

/// Train (or finetune) one head end to end: split, fit, measure, refit on all.
#[allow(clippy::too_many_arguments)]
fn fit_head(
    rows: &[TrainRow],
    idx: &[usize],
    targets: &[f32],
    init: Option<&[f32]>,
    n_features: usize,
    opts: &TrainOptions,
    lr0: f32,
    rng: &mut Rng,
    label: &str,
    job: &JobHandle,
) -> (Vec<f32>, HeadMetrics) {
    let (train_pos, hold_pos) = split_stratified(rows, idx, targets, opts.holdout, rng);
    let sub = |positions: &[usize]| -> (Vec<usize>, Vec<f32>) {
        (positions.iter().map(|&k| idx[k]).collect(), positions.iter().map(|&k| targets[k]).collect())
    };
    let mut metrics = HeadMetrics::default();
    if !hold_pos.is_empty() {
        let (tr_idx, tr_t) = sub(&train_pos);
        let sw = balanced_weights(rows, &tr_idx, &tr_t);
        let epochs = opts.epochs.max(1);
        let w = train_head(rows, &tr_idx, &tr_t, &sw, init, n_features, epochs, lr0, rng, |e| {
            job.progress(e as u64, epochs as u64 * 2, format!("{label}: epoch {e}/{epochs} (holdout fit)"));
        });
        let preds: Vec<(bool, bool)> = hold_pos
            .iter()
            .map(|&k| (builtin::sigmoid(builtin::dot(&w, &rows[idx[k]].x)) >= 0.5, targets[k] > 0.5))
            .collect();
        metrics = head_metrics(&preds);
        metrics.n_train = tr_idx.len();
    }
    // Final weights: fit on everything.
    let sw = balanced_weights(rows, idx, targets);
    let epochs = opts.epochs.max(1);
    let w = train_head(rows, idx, targets, &sw, init, n_features, epochs, lr0, rng, |e| {
        job.progress(epochs as u64 + e as u64, epochs as u64 * 2, format!("{label}: epoch {e}/{epochs} (final fit)"));
    });
    metrics.positives = targets.iter().filter(|&&t| t > 0.5).count();
    if metrics.n_train == 0 {
        metrics.n_train = idx.len();
    }
    (w, metrics)
}

pub fn train_builtin(db: &Db, job: &JobHandle, models_dir: &Path, opts: &TrainOptions) -> Result<(i64, TrainMetrics)> {
    let t0 = Instant::now();
    let name = if opts.name.trim().is_empty() { format!("model-{}", stamp()) } else { opts.name.trim().to_string() };

    // 1. Parent (finetune) — decide feature config and idf source.
    let parent: Option<(db::ModelInfo, BuiltinModel)> = match opts.parent_model {
        Some(pid) => {
            let info = db::get_model(&db.lock(), pid)?;
            if info.kind != "builtin" {
                return Err(GccfError::msg("only built-in models can be finetuned in-process; use the external trainer with {parent_model} for others"));
            }
            let model = BuiltinModel::load(Path::new(&info.path)).map_err(|e| GccfError::msg(e.to_string()))?;
            Some((info, model))
        }
        None => None,
    };
    let cfg: FeatureConfig = parent.as_ref().map(|(_, m)| m.header.features.clone()).unwrap_or_default();
    let n_features = cfg.n_features();

    // 2. Load rows (texts) under the lock, then featurise outside it.
    job.progress(0, 0, "loading annotated samples");
    struct Raw {
        sample_id: i64,
        text: String,
        label: Option<u8>,
        tags: Vec<String>,
    }
    let (raws, disagreements): (Vec<Raw>, HashMap<i64, bool>) = {
        let conn = db.lock();
        let mut raws = Vec::new();
        db::for_each_sample_export(&conn, &opts.session_ids, &["good", "bad", "skip"], |row| {
            if opts.human_only && row.decided_by.as_deref() == Some("classifier") {
                return Ok(());
            }
            raws.push(Raw {
                sample_id: row.sample_id,
                text: row.text,
                label: match row.status.as_str() {
                    "good" => Some(1),
                    "bad" => Some(0),
                    _ => None,
                },
                tags: row.tags,
            });
            Ok(())
        })?;
        (raws, db::eval_disagreement_samples(&conn)?)
    };
    let labelled = raws.iter().filter(|r| r.label.is_some()).count();
    let good = raws.iter().filter(|r| r.label == Some(1)).count();
    let bad = labelled - good;
    if labelled < 20 || good == 0 || bad == 0 {
        return Err(GccfError::msg(format!(
            "need at least 20 labelled samples with both classes (have {good} good, {bad} bad) — keep annotating"
        )));
    }

    job.progress(0, raws.len() as u64, format!("featurising {} samples", raws.len()));
    let feedback_weight = opts.feedback_weight.max(1.0) as f32;
    let mut rows: Vec<TrainRow> = {
        use rayon::prelude::*;
        raws.into_par_iter()
            .map(|r| TrainRow {
                label: r.label,
                tags: r.tags,
                weight: if disagreements.get(&r.sample_id).copied().unwrap_or(false) { feedback_weight } else { 1.0 },
                x: builtin::raw_features(&r.text, &cfg),
            })
            .collect()
    };
    let feedback_rows = rows.iter().filter(|r| r.weight > 1.0).count();
    if job.is_cancelled() {
        return Err(GccfError::msg("cancelled"));
    }

    // 3. idf: inherit from the parent when finetuning, else from this corpus.
    let idf: Vec<f32> = match &parent {
        Some((_, m)) => m.idf.clone(),
        None => {
            let mut df = vec![0u32; n_features];
            for r in &rows {
                for &(i, _) in &r.x {
                    df[i as usize] += 1;
                }
            }
            let n = rows.len() as f32;
            df.iter().map(|&d| ((n + 1.0) / (d as f32 + 1.0)).ln() + 1.0).collect()
        }
    };
    for r in rows.iter_mut() {
        builtin::finish_features(&mut r.x, &idf);
    }

    let mut rng = Rng::new(opts.seed);
    let lr0 = if parent.is_some() { 0.15 } else { 0.5 };
    let mut metrics = TrainMetrics {
        rows: rows.len(),
        labelled,
        good,
        bad,
        feedback_rows,
        warm_start: parent.is_some(),
        epochs: opts.epochs,
        ..Default::default()
    };

    // 4. Verdict head.
    let v_idx: Vec<usize> = rows.iter().enumerate().filter(|(_, r)| r.label.is_some()).map(|(i, _)| i).collect();
    let v_t: Vec<f32> = v_idx.iter().map(|&i| rows[i].label.unwrap() as f32).collect();
    let (verdict_w, vm) = fit_head(
        &rows,
        &v_idx,
        &v_t,
        parent.as_ref().map(|(_, m)| m.verdict.as_slice()),
        n_features,
        opts,
        lr0,
        &mut rng,
        "good/bad",
        job,
    );
    metrics.verdict = vm;
    if job.is_cancelled() {
        return Err(GccfError::msg("cancelled"));
    }

    // 5. Tag heads.
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for r in &rows {
        for t in &r.tags {
            *counts.entry(t.as_str()).or_default() += 1;
        }
    }
    let mut tag_names: Vec<String> = counts
        .iter()
        .filter(|(_, &c)| c >= opts.min_tag_count.max(2) && c < rows.len())
        .map(|(n, _)| n.to_string())
        .collect();
    tag_names.sort();
    let all_idx: Vec<usize> = (0..rows.len()).collect();
    let mut tag_heads = Vec::with_capacity(tag_names.len());
    for (ti, tag) in tag_names.iter().enumerate() {
        let targets: Vec<f32> = rows.iter().map(|r| if r.tags.iter().any(|t| t == tag) { 1.0 } else { 0.0 }).collect();
        let init = parent.as_ref().and_then(|(_, m)| {
            m.header.tag_names.iter().position(|n| n == tag).map(|p| m.tags[p].as_slice())
        });
        let (w, hm) = fit_head(&rows, &all_idx, &targets, init, n_features, opts, lr0, &mut rng, &format!("tag {tag} ({}/{})", ti + 1, tag_names.len()), job);
        tag_heads.push(w);
        metrics.tags.push(TagMetrics {
            name: tag.clone(),
            count: counts[tag.as_str()],
            metrics: hm,
        });
        if job.is_cancelled() {
            return Err(GccfError::msg("cancelled"));
        }
    }

    // 6. Persist.
    metrics.elapsed_ms = t0.elapsed().as_millis();
    let metrics_json = serde_json::to_value(&metrics)?;
    let path: PathBuf = models_dir.join(format!("{}-{}.gccfm", slug(&name), stamp()));
    let model = BuiltinModel {
        header: ModelHeader {
            name: name.clone(),
            created_at: db::now_iso(),
            features: cfg,
            tag_names: tag_names.clone(),
            tag_threshold: 0.5,
            train_rows: rows.len(),
            metrics: metrics_json.clone(),
        },
        idf,
        verdict: verdict_w,
        tags: tag_heads,
    };
    model.save(&path).map_err(|e| GccfError::msg(e.to_string()))?;
    let id = db::insert_model(
        &db.lock(),
        &db::NewModel {
            name: &name,
            kind: "builtin",
            path: &path.to_string_lossy(),
            command: None,
            parent_id: parent.as_ref().map(|(i, _)| i.id),
            train_rows: rows.len() as i64,
            metrics: &metrics_json,
            tag_names: &tag_names,
            notes: opts.notes.as_deref(),
        },
    )?;
    let _ = DENSE;
    Ok((id, metrics))
}

// ---------------------------------------------------------------------------
// External trainer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ExternalTrainOptions {
    pub name: String,
    #[serde(default)]
    pub session_ids: Vec<i64>,
    #[serde(default = "d_true")]
    pub human_only: bool,
    /// e.g. `py -3 classifier/train_classifier.py {data} --out {model}`
    pub command_template: String,
    #[serde(default)]
    pub cwd: Option<String>,
    /// e.g. `py -3 classifier/worker.py --model {model}` — stored as the
    /// model's serving command.
    pub worker_template: String,
    #[serde(default = "d_ext")]
    pub model_ext: String,
    #[serde(default)]
    pub parent_model: Option<i64>,
    #[serde(default)]
    pub notes: Option<String>,
}

const LOG_KEEP: usize = 400;

fn push_log(log: &Arc<Mutex<Vec<String>>>, line: String) {
    let mut g = log.lock().unwrap();
    g.push(line);
    if g.len() > LOG_KEEP * 2 {
        let drop = g.len() - LOG_KEEP;
        g.drain(..drop);
    }
}

pub fn train_external(db: &Db, job: &JobHandle, models_dir: &Path, opts: &ExternalTrainOptions) -> Result<i64> {
    let name = if opts.name.trim().is_empty() { format!("model-{}", stamp()) } else { opts.name.trim().to_string() };
    let base = format!("{}-{}", slug(&name), stamp());
    std::fs::create_dir_all(models_dir)?;
    let data_path = models_dir.join(format!("{base}.train.jsonl"));
    let ext = opts.model_ext.trim().trim_start_matches('.');
    let model_path = models_dir.join(if ext.is_empty() { base.clone() } else { format!("{base}.{ext}") });

    // 1. Export the training set (with evaluation feedback columns).
    job.progress(0, 0, "exporting training data");
    let stats = crate::export::export_training_jsonl(&db.lock(), &data_path, &opts.session_ids, false, true, opts.human_only)?;
    let parent_path = match opts.parent_model {
        Some(pid) => Some(db::get_model(&db.lock(), pid)?.path),
        None => None,
    };

    // 2. Build and spawn the command.
    let cmdline = opts
        .command_template
        .replace("{data}", &format!("\"{}\"", data_path.display()))
        .replace("{model}", &format!("\"{}\"", model_path.display()))
        .replace("{parent_model}", &format!("\"{}\"", parent_path.clone().unwrap_or_default()));
    let mut parts = split_command_line(&cmdline);
    if parts.is_empty() {
        return Err(GccfError::msg("empty training command"));
    }
    let program = parts.remove(0);
    let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![format!("$ {cmdline}"), format!("exported {} rows ({} good / {} bad)", stats.rows, stats.good, stats.bad)]));
    let mut cmd = std::process::Command::new(&program);
    cmd.args(&parts)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("GCCF_DATA", &data_path)
        .env("GCCF_MODEL_OUT", &model_path);
    if let Some(p) = &parent_path {
        cmd.env("GCCF_PARENT_MODEL", p);
    }
    if let Some(dir) = &opts.cwd {
        if !dir.trim().is_empty() {
            cmd.current_dir(dir);
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = cmd.spawn().map_err(|e| GccfError::msg(format!("spawning `{program}`: {e}")))?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let metrics_line: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));
    let readers: Vec<std::thread::JoinHandle<()>> = [(Box::new(stdout) as Box<dyn std::io::Read + Send>, "out"), (Box::new(stderr), "err")]
        .into_iter()
        .map(|(stream, tag)| {
            let log = log.clone();
            let metrics_line = metrics_line.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stream).lines().map_while(|l| l.ok()) {
                    if let Some(rest) = line.trim().strip_prefix("{\"gccf_metrics\":") {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&format!("{{\"gccf_metrics\":{rest}")) {
                            *metrics_line.lock().unwrap() = v.get("gccf_metrics").cloned();
                        }
                    }
                    push_log(&log, if tag == "err" { format!("! {line}") } else { line });
                }
            })
        })
        .collect();

    // 3. Pump the log into the job until the process exits.
    let status = loop {
        if let Some(st) = child.try_wait()? {
            break st;
        }
        if job.is_cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(GccfError::msg("training cancelled"));
        }
        {
            let g = log.lock().unwrap();
            let tail: Vec<&String> = g.iter().rev().take(LOG_KEEP).collect::<Vec<_>>().into_iter().rev().collect();
            job.set_payload(serde_json::json!({ "log": tail }));
            job.progress(0, 0, g.last().cloned().unwrap_or_default());
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    };
    for r in readers {
        let _ = r.join();
    }
    let tail: Vec<String> = log.lock().unwrap().iter().rev().take(LOG_KEEP).cloned().collect::<Vec<_>>().into_iter().rev().collect();
    job.set_payload(serde_json::json!({ "log": tail }));
    if !status.success() {
        return Err(GccfError::msg(format!("training command exited with {status}; see log")));
    }
    if !model_path.exists() {
        return Err(GccfError::msg(format!("training command finished but did not write {}", model_path.display())));
    }

    // 4. Register.
    let worker_cmd = opts.worker_template.replace("{model}", &format!("\"{}\"", model_path.display()));
    let mut metrics = metrics_line.lock().unwrap().clone().unwrap_or(serde_json::Value::Null);
    if metrics.is_null() {
        metrics = serde_json::json!({ "log_tail": tail.iter().rev().take(30).cloned().collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>() });
    }
    if let Some(obj) = metrics.as_object_mut() {
        obj.insert("rows".into(), serde_json::json!(stats.rows));
        obj.insert("good".into(), serde_json::json!(stats.good));
        obj.insert("bad".into(), serde_json::json!(stats.bad));
        obj.insert("data_path".into(), serde_json::json!(data_path.display().to_string()));
    }
    let id = db::insert_model(
        &db.lock(),
        &db::NewModel {
            name: &name,
            kind: "external",
            path: &model_path.to_string_lossy(),
            command: Some(&worker_cmd),
            parent_id: opts.parent_model,
            train_rows: stats.rows,
            metrics: &metrics,
            tag_names: &[],
            notes: opts.notes.as_deref(),
        },
    )?;
    Ok(id)
}

/// Unique tag names present in a set of rows — exposed for the UI so it can
/// warn before training when the tag vocabulary is thin.
pub fn distinct_tags(rows: &[Vec<String>]) -> usize {
    rows.iter().flatten().collect::<HashSet<_>>().len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn trains_finetunes_and_predicts() {
        // Synthetic corpus: prose-like "good" rows vs. symbol junk "bad" rows.
        let dir = tempfile::tempdir().unwrap();
        let mut fh = std::fs::File::create(dir.path().join("mix.jsonl")).unwrap();
        for i in 0..160 {
            if i % 2 == 0 {
                writeln!(fh, "{{\"text\": \"The committee reviewed proposal {i} and decided that the evidence supports waiting for more data before a final decision is made about the budget.\"}}").unwrap();
            } else {
                writeln!(fh, "{{\"text\": \"### ### ||| {i} ||| @@@ ### $$$ %%% ### ||| @@@ &&& *** ### ||| {i}{i} ||| ### @@@\"}}").unwrap();
            }
        }
        drop(fh);
        let db = Arc::new(Db::open_memory().unwrap());
        let jobs = Arc::new(crate::jobs::Jobs::new());
        let job = jobs.start(None, "x", 0, "", serde_json::Value::Null);
        let root_id = db::insert_root(&db.lock(), &dir.path().to_string_lossy(), None).unwrap();
        crate::scan::scan_root(&db, &job, root_id, dir.path()).unwrap();
        let sid = db::insert_session(
            &db.lock(),
            &db::NewSession { root_id, name: "s", files_requested: 1, samples_per_file: 160, lines_per_sample: 1, seed: 1, format_filter: &[], exclude_reviewed_files: false },
        )
        .unwrap();
        crate::session::build_session(db.clone(), job.clone(), sid, crate::session::BuildConfig { threads: 2, max_bytes_per_sample: 4096, max_stream_bytes: 0 }).unwrap();

        // Label every sample by content; tag the prose rows "english".
        {
            let conn = db.lock();
            let tag = db::create_tag(&conn, "english").unwrap();
            let mut stmt = conn.prepare("SELECT id, text FROM samples WHERE session_id = ?1").unwrap();
            let rows: Vec<(i64, String)> = stmt.query_map([sid], |r| Ok((r.get(0)?, r.get(1)?))).unwrap().map(|r| r.unwrap()).collect();
            assert_eq!(rows.len(), 160);
            for (id, text) in rows {
                let good = text.contains("committee");
                if good {
                    db::set_sample_tag(&conn, id, tag, true, "human").unwrap();
                }
                db::decide(&conn, id, if good { "good" } else { "bad" }, "human", None, None, None).unwrap();
            }
        }

        let models_dir = dir.path().join("models");
        let opts = TrainOptions {
            name: "Test Model".into(),
            session_ids: vec![],
            human_only: true,
            holdout: 0.2,
            min_tag_count: 5,
            epochs: 8,
            parent_model: None,
            feedback_weight: 3.0,
            seed: 1,
            notes: None,
        };
        let (model_id, m) = train_builtin(&db, &job, &models_dir, &opts).unwrap();
        assert_eq!(m.rows, 160);
        assert!(m.verdict.accuracy.unwrap() >= 0.9, "holdout accuracy {:?}", m.verdict.accuracy);
        assert_eq!(m.tags.len(), 1);
        assert!(m.tags[0].metrics.f1.unwrap() >= 0.9);

        let info = db::get_model(&db.lock(), model_id).unwrap();
        assert_eq!(info.kind, "builtin");
        let model = BuiltinModel::load(Path::new(&info.path)).unwrap();
        let (v, s, t) = model.predict("The board reviewed the evidence and decided to wait for more data before the final decision.");
        assert_eq!(v, "good");
        assert!(s > 0.5);
        assert_eq!(t, vec!["english".to_string()]);
        let (v, _, _) = model.predict("@@@ ### $$$ ||| %%% *** ### ||| @@@ &&&");
        assert_eq!(v, "bad");

        // Finetune from the parent: same feature space, warm start, must still work.
        let (child_id, m2) = train_builtin(&db, &job, &models_dir, &TrainOptions { name: "child".into(), parent_model: Some(model_id), epochs: 3, ..opts.clone() }).unwrap();
        assert!(m2.warm_start);
        let child = db::get_model(&db.lock(), child_id).unwrap();
        assert_eq!(child.parent_id, Some(model_id));
        let cm = BuiltinModel::load(Path::new(&child.path)).unwrap();
        assert_eq!(cm.idf, model.idf, "finetune keeps the parent's feature space");
        assert_eq!(cm.predict("The committee decided to wait for data.").0, "good");
    }

    #[test]
    fn slugs() {
        assert_eq!(slug("My Model v2!"), "my-model-v2");
        assert_eq!(slug("   "), "model");
    }
}
