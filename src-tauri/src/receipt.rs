//! Corpora ingestion receipt. This is GCCF's output artifact: a versioned JSON
//! document that tells a downstream ingester which files to include, which
//! records inside them to drop, and what the reviewer said about the rest.
//! Schema documented in `docs/receipt-schema.md`.

use crate::db::{self};
use crate::error::Result;
use rusqlite::{params_from_iter, Connection};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const RECEIPT_SCHEMA: &str = "gccf.receipt/1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptOptions {
    /// Empty = every session.
    #[serde(default)]
    pub session_ids: Vec<i64>,
    /// "summary" | "files" | "full"
    #[serde(default = "default_verbosity")]
    pub verbosity: String,
    #[serde(default = "default_include_ratio")]
    pub include_min_good_ratio: f64,
    #[serde(default = "default_exclude_ratio")]
    pub exclude_max_good_ratio: f64,
    #[serde(default = "default_min_decided")]
    pub min_decided: i64,
    #[serde(default)]
    pub include_unsampled_files: bool,
    #[serde(default)]
    pub skip_counts_as_bad: bool,
    #[serde(default = "default_true")]
    pub include_classifier_decisions: bool,
    /// Only honoured with verbosity = "full".
    #[serde(default)]
    pub include_text: bool,
    #[serde(default)]
    pub note: Option<String>,
}

fn default_verbosity() -> String {
    "files".into()
}
fn default_include_ratio() -> f64 {
    0.8
}
fn default_exclude_ratio() -> f64 {
    0.5
}
fn default_min_decided() -> i64 {
    1
}
fn default_true() -> bool {
    true
}

impl Default for ReceiptOptions {
    fn default() -> Self {
        ReceiptOptions {
            session_ids: Vec::new(),
            verbosity: default_verbosity(),
            include_min_good_ratio: default_include_ratio(),
            exclude_max_good_ratio: default_exclude_ratio(),
            min_decided: default_min_decided(),
            include_unsampled_files: false,
            skip_counts_as_bad: false,
            include_classifier_decisions: true,
            include_text: false,
            note: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Counts {
    pub samples: i64,
    pub good: i64,
    pub bad: i64,
    pub skip: i64,
    pub pending: i64,
    pub human: i64,
    pub classifier: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReceiptSample {
    pub id: i64,
    pub session_id: i64,
    pub locator: serde_json::Value,
    pub verdict: String,
    pub source: Option<String>,
    pub tags: Vec<String>,
    pub content_hash: String,
    pub decided_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReceiptFile {
    pub root_id: i64,
    pub path: String,
    pub abs_path: String,
    pub format: String,
    pub kind: String,
    pub compression: String,
    pub size_bytes: i64,
    pub mtime: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub row_count: Option<i64>,
    pub verdict: String,
    pub good_ratio: Option<f64>,
    pub counts: Counts,
    pub tags: BTreeMap<String, i64>,
    /// Locators of records the reviewer marked bad: a downstream ingester can
    /// drop exactly these rows even when the file as a whole is included.
    pub exclusions: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub samples: Option<Vec<ReceiptSample>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReceiptRoot {
    pub id: i64,
    pub path: String,
    pub scanned_at: Option<String>,
    pub file_count: i64,
    pub total_bytes: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReceiptSession {
    pub id: i64,
    pub root_id: i64,
    pub name: String,
    pub created_at: String,
    pub files_requested: i64,
    pub files_selected: i64,
    pub samples_per_file: i64,
    pub lines_per_sample: i64,
    pub seed: i64,
    pub format_filter: Vec<String>,
    pub counts: db::StatusCounts,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReceiptTag {
    pub name: String,
    pub builtin: bool,
    pub sample_count: i64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ReceiptSummary {
    pub files_include: i64,
    pub files_exclude: i64,
    pub files_review: i64,
    pub files_unsampled: i64,
    pub bytes_include: i64,
    pub bytes_exclude: i64,
    pub bytes_review: i64,
    pub samples: Counts,
    pub exclusions: i64,
    pub tags: BTreeMap<String, i64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Receipt {
    pub schema: String,
    pub generated_at: String,
    pub generator: serde_json::Value,
    pub note: Option<String>,
    pub policy: serde_json::Value,
    pub tags: Vec<ReceiptTag>,
    pub roots: Vec<ReceiptRoot>,
    pub sessions: Vec<ReceiptSession>,
    pub summary: ReceiptSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files: Option<Vec<ReceiptFile>>,
}

struct SampleRow {
    id: i64,
    session_id: i64,
    file_id: i64,
    locator: serde_json::Value,
    status: String,
    decided_by: Option<String>,
    decided_at: Option<String>,
    content_hash: String,
    text: Option<String>,
}

pub fn build_receipt(conn: &Connection, opts: &ReceiptOptions) -> Result<Receipt> {
    let verbosity = opts.verbosity.as_str();
    let full = verbosity == "full";
    let with_files = verbosity != "summary";

    // Sessions in scope.
    let all_sessions = db::list_sessions(conn)?;
    let sessions: Vec<db::SessionInfo> = if opts.session_ids.is_empty() {
        all_sessions
    } else {
        all_sessions.into_iter().filter(|s| opts.session_ids.contains(&s.id)).collect()
    };
    let session_ids: Vec<i64> = sessions.iter().map(|s| s.id).collect();
    let root_ids: std::collections::BTreeSet<i64> = sessions.iter().map(|s| s.root_id).collect();

    let roots: Vec<ReceiptRoot> = db::list_roots(conn)?
        .into_iter()
        .filter(|r| root_ids.contains(&r.id))
        .map(|r| ReceiptRoot {
            id: r.id,
            path: r.path,
            scanned_at: r.scanned_at,
            file_count: r.file_count,
            total_bytes: r.total_bytes,
        })
        .collect();

    // Samples in scope, grouped by file.
    let mut by_file: BTreeMap<i64, Vec<SampleRow>> = BTreeMap::new();
    if !session_ids.is_empty() {
        let ph = session_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT id, session_id, file_id, locator_json, status, decided_by, decided_at, content_hash, {} FROM samples WHERE session_id IN ({ph}) ORDER BY file_id, ordinal",
            if full && opts.include_text { "text" } else { "NULL" }
        );
        let mut stmt = conn.prepare(&sql)?;
        let args: Vec<rusqlite::types::Value> = session_ids.iter().map(|id| (*id).into()).collect();
        let rows = stmt.query_map(params_from_iter(args.iter()), |r| {
            let loc: String = r.get(3)?;
            Ok(SampleRow {
                id: r.get(0)?,
                session_id: r.get(1)?,
                file_id: r.get(2)?,
                locator: serde_json::from_str(&loc).unwrap_or(serde_json::Value::Null),
                status: r.get(4)?,
                decided_by: r.get(5)?,
                decided_at: r.get(6)?,
                content_hash: r.get(7)?,
                text: r.get(8)?,
            })
        })?;
        for row in rows {
            let row = row?;
            by_file.entry(row.file_id).or_default().push(row);
        }
    }

    let mut summary = ReceiptSummary::default();
    let mut files: Vec<ReceiptFile> = Vec::new();
    let file_ids: Vec<i64> = by_file.keys().copied().collect();
    let file_rows = db::get_files(conn, &file_ids)?;
    let file_map: BTreeMap<i64, db::FileRow> = file_rows.into_iter().map(|f| (f.id, f)).collect();

    for (file_id, samples) in &by_file {
        let Some(f) = file_map.get(file_id) else { continue };
        let mut counts = Counts::default();
        let mut tags: BTreeMap<String, i64> = BTreeMap::new();
        let mut exclusions = Vec::new();
        let mut out_samples = Vec::new();
        for s in samples {
            counts.samples += 1;
            let effective = if !opts.include_classifier_decisions && s.decided_by.as_deref() == Some("classifier") {
                "pending"
            } else {
                s.status.as_str()
            };
            match effective {
                "good" => counts.good += 1,
                "bad" => counts.bad += 1,
                "skip" => counts.skip += 1,
                _ => counts.pending += 1,
            }
            match s.decided_by.as_deref() {
                Some("human") => counts.human += 1,
                Some("classifier") => counts.classifier += 1,
                _ => {}
            }
            let names = db::sample_tag_names(conn, s.id)?;
            for n in &names {
                *tags.entry(n.clone()).or_default() += 1;
                *summary.tags.entry(n.clone()).or_default() += 1;
            }
            if effective == "bad" || (opts.skip_counts_as_bad && effective == "skip") {
                exclusions.push(s.locator.clone());
            }
            if full {
                out_samples.push(ReceiptSample {
                    id: s.id,
                    session_id: s.session_id,
                    locator: s.locator.clone(),
                    verdict: effective.to_string(),
                    source: s.decided_by.clone(),
                    tags: names,
                    content_hash: s.content_hash.clone(),
                    decided_at: s.decided_at.clone(),
                    text: s.text.clone(),
                });
            }
        }
        let bad_like = counts.bad + if opts.skip_counts_as_bad { counts.skip } else { 0 };
        let decided = counts.good + bad_like;
        let good_ratio = if decided > 0 { Some(counts.good as f64 / decided as f64) } else { None };
        let verdict = match good_ratio {
            Some(r) if decided >= opts.min_decided && r >= opts.include_min_good_ratio => "include",
            Some(r) if decided >= opts.min_decided && r <= opts.exclude_max_good_ratio => "exclude",
            _ => "review",
        };
        match verdict {
            "include" => {
                summary.files_include += 1;
                summary.bytes_include += f.size_bytes;
            }
            "exclude" => {
                summary.files_exclude += 1;
                summary.bytes_exclude += f.size_bytes;
            }
            _ => {
                summary.files_review += 1;
                summary.bytes_review += f.size_bytes;
            }
        }
        summary.samples.samples += counts.samples;
        summary.samples.good += counts.good;
        summary.samples.bad += counts.bad;
        summary.samples.skip += counts.skip;
        summary.samples.pending += counts.pending;
        summary.samples.human += counts.human;
        summary.samples.classifier += counts.classifier;
        summary.exclusions += exclusions.len() as i64;
        if with_files {
            files.push(ReceiptFile {
                root_id: f.root_id,
                path: f.rel_path.clone(),
                abs_path: f.abs_path().to_string_lossy().to_string(),
                format: f.format.clone(),
                kind: f.kind.clone(),
                compression: f.compression.clone(),
                size_bytes: f.size_bytes,
                mtime: f.mtime,
                row_count: f.row_count,
                verdict: verdict.to_string(),
                good_ratio,
                counts,
                tags,
                exclusions,
                samples: if full { Some(out_samples) } else { None },
            });
        }
    }

    if opts.include_unsampled_files {
        for root_id in &root_ids {
            let mut stmt = conn.prepare(
                "SELECT f.id, f.rel_path, f.size_bytes, f.mtime, f.format, f.kind, f.compression, f.row_count, r.path
                 FROM files f JOIN roots r ON r.id = f.root_id
                 WHERE f.root_id = ?1 AND f.missing = 0 AND NOT EXISTS (SELECT 1 FROM samples s WHERE s.file_id = f.id)
                 ORDER BY f.rel_path",
            )?;
            let rows = stmt.query_map([root_id], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, i64>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, String>(5)?,
                    r.get::<_, String>(6)?,
                    r.get::<_, Option<i64>>(7)?,
                    r.get::<_, String>(8)?,
                ))
            })?;
            for row in rows {
                let (_id, rel, size, mtime, format, kind, compression, row_count, root_path) = row?;
                summary.files_unsampled += 1;
                if with_files {
                    files.push(ReceiptFile {
                        root_id: *root_id,
                        abs_path: std::path::Path::new(&root_path).join(&rel).to_string_lossy().to_string(),
                        path: rel,
                        format,
                        kind,
                        compression,
                        size_bytes: size,
                        mtime,
                        row_count,
                        verdict: "unsampled".into(),
                        good_ratio: None,
                        counts: Counts::default(),
                        tags: BTreeMap::new(),
                        exclusions: Vec::new(),
                        samples: None,
                    });
                }
            }
        }
    }

    let tags: Vec<ReceiptTag> = db::list_tags(conn)?
        .into_iter()
        .map(|t| ReceiptTag {
            sample_count: summary.tags.get(&t.name).copied().unwrap_or(0),
            name: t.name,
            builtin: t.builtin,
        })
        .collect();

    Ok(Receipt {
        schema: RECEIPT_SCHEMA.into(),
        generated_at: db::now_iso(),
        generator: serde_json::json!({ "name": "gccf", "version": env!("CARGO_PKG_VERSION") }),
        note: opts.note.clone(),
        policy: serde_json::json!({
            "verbosity": verbosity,
            "include_min_good_ratio": opts.include_min_good_ratio,
            "exclude_max_good_ratio": opts.exclude_max_good_ratio,
            "min_decided": opts.min_decided,
            "skip_counts_as_bad": opts.skip_counts_as_bad,
            "include_classifier_decisions": opts.include_classifier_decisions,
            "include_unsampled_files": opts.include_unsampled_files,
            "file_verdicts": ["include", "exclude", "review", "unsampled"],
            "locator_types": {
                "row": "absolute row index in a parquet file",
                "bytes": "line-aligned byte range [byte_start, byte_end) in an uncompressed file; n_lines lines",
                "stream_lines": "0-based line window inside a gzip/zstd stream after decompression"
            }
        }),
        tags,
        roots,
        sessions: sessions
            .into_iter()
            .map(|s| ReceiptSession {
                id: s.id,
                root_id: s.root_id,
                name: s.name,
                created_at: s.created_at,
                files_requested: s.files_requested,
                files_selected: s.files_selected,
                samples_per_file: s.samples_per_file,
                lines_per_sample: s.lines_per_sample,
                seed: s.seed,
                format_filter: s.format_filter,
                counts: s.counts,
            })
            .collect(),
        summary,
        files: if with_files { Some(files) } else { None },
    })
}

pub fn write_receipt(conn: &Connection, path: &std::path::Path, opts: &ReceiptOptions) -> Result<ReceiptSummary> {
    let receipt = build_receipt(conn, opts)?;
    let json = serde_json::to_vec_pretty(&receipt)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, json)?;
    Ok(receipt.summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Db;
    use std::io::Write;
    use std::sync::Arc;

    /// Scan → session → decisions → receipt + training export, end to end.
    #[test]
    fn receipt_and_export_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        for f in 0..3 {
            let mut fh = std::fs::File::create(dir.path().join(format!("s{f}.jsonl"))).unwrap();
            for i in 0..100 {
                writeln!(fh, "{{\"id\": {i}, \"text\": \"document {i} of shard {f}\"}}").unwrap();
            }
        }
        let db = Arc::new(Db::open_memory().unwrap());
        let jobs = Arc::new(crate::jobs::Jobs::new());
        let root_id = db::insert_root(&db.lock(), &dir.path().to_string_lossy(), None).unwrap();
        crate::scan::scan_root(&db, &jobs.start(None, "scan", 0, "", serde_json::Value::Null), root_id, dir.path()).unwrap();
        let sid = db::insert_session(
            &db.lock(),
            &db::NewSession { root_id, name: "r", files_requested: 3, samples_per_file: 10, lines_per_sample: 1, seed: 5, format_filter: &[], exclude_reviewed_files: true },
        )
        .unwrap();
        crate::session::build_session(
            db.clone(),
            jobs.start(None, "session", 0, "", serde_json::Value::Null),
            sid,
            crate::session::BuildConfig { threads: 2, max_bytes_per_sample: 4096, max_stream_bytes: 0 },
        )
        .unwrap();

        // Decide: file 0 all good, file 1 mostly bad, file 2 untouched.
        {
            let conn = db.lock();
            let tag = db::create_tag(&conn, "english").unwrap();
            let mut stmt = conn
                .prepare("SELECT s.id, f.rel_path FROM samples s JOIN files f ON f.id = s.file_id WHERE s.session_id = ?1 ORDER BY s.id")
                .unwrap();
            let rows: Vec<(i64, String)> = stmt.query_map([sid], |r| Ok((r.get(0)?, r.get(1)?))).unwrap().map(|r| r.unwrap()).collect();
            for (i, (id, rel)) in rows.iter().enumerate() {
                if rel.starts_with("s0") {
                    db::set_sample_tag(&conn, *id, tag, true, "human").unwrap();
                    db::decide(&conn, *id, "good", "human", None, None, Some(500)).unwrap();
                } else if rel.starts_with("s1") {
                    db::decide(&conn, *id, if i % 5 == 0 { "skip" } else { "bad" }, "human", None, None, None).unwrap();
                }
            }
        }

        let opts = ReceiptOptions { verbosity: "full".into(), include_unsampled_files: true, ..Default::default() };
        let receipt = build_receipt(&db.lock(), &opts).unwrap();
        assert_eq!(receipt.schema, RECEIPT_SCHEMA);
        assert_eq!(receipt.summary.files_include, 1);
        assert_eq!(receipt.summary.files_exclude, 1);
        assert_eq!(receipt.summary.files_review, 1); // s2: sampled but nothing decided
        assert_eq!(receipt.summary.samples.samples, 30);
        assert_eq!(receipt.summary.tags.get("english"), Some(&10));
        let files = receipt.files.as_ref().unwrap();
        let inc = files.iter().find(|f| f.verdict == "include").unwrap();
        assert!(inc.path.starts_with("s0"));
        assert_eq!(inc.counts.good, 10);
        assert!(inc.exclusions.is_empty());
        let exc = files.iter().find(|f| f.verdict == "exclude").unwrap();
        assert_eq!(exc.exclusions.len(), exc.counts.bad as usize);
        assert_eq!(exc.samples.as_ref().unwrap().len(), 10);
        assert!(exc.samples.as_ref().unwrap()[0].text.is_none(), "text only with include_text");

        // Files verbosity drops samples; summary drops files.
        let r2 = build_receipt(&db.lock(), &ReceiptOptions { verbosity: "files".into(), ..Default::default() }).unwrap();
        assert!(r2.files.as_ref().unwrap().iter().all(|f| f.samples.is_none()));
        let r3 = build_receipt(&db.lock(), &ReceiptOptions { verbosity: "summary".into(), ..Default::default() }).unwrap();
        assert!(r3.files.is_none());

        // Write to disk and parse back.
        let out = dir.path().join("receipt.json");
        write_receipt(&db.lock(), &out, &opts).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
        assert_eq!(v["schema"], "gccf.receipt/1");
        assert_eq!(v["files"].as_array().unwrap().len(), 3);

        // Training export: labelled rows only by default.
        let jsonl = dir.path().join("train.jsonl");
        let st = crate::export::export_training_jsonl(&db.lock(), &jsonl, &[], false, false, true).unwrap();
        assert_eq!(st.good, 10);
        assert_eq!(st.bad, 8);
        assert_eq!(st.rows, 18);
        let first: serde_json::Value = serde_json::from_str(std::fs::read_to_string(&jsonl).unwrap().lines().next().unwrap()).unwrap();
        assert!(first["label"].is_number());
        assert!(first["text"].is_string());
        let st2 = crate::export::export_training_jsonl(&db.lock(), &jsonl, &[], true, true, true).unwrap();
        assert_eq!(st2.rows, 30);
    }
}
