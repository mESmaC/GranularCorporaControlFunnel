//! Classifier training-data export: one JSON object per line with the sample
//! text, the human/classifier verdict, tags, and provenance. This is the file
//! `classifier/train_classifier.py` consumes.

use crate::db;
use crate::error::Result;
use rusqlite::Connection;
use serde::Serialize;
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Default)]
pub struct ExportStats {
    pub rows: i64,
    pub bytes: u64,
    pub good: i64,
    pub bad: i64,
    pub skip: i64,
    pub pending: i64,
    pub human: i64,
    pub classifier: i64,
}

pub fn export_training_jsonl(
    conn: &Connection,
    path: &Path,
    session_ids: &[i64],
    include_pending: bool,
    include_skipped: bool,
    human_only: bool,
) -> Result<ExportStats> {
    let mut statuses: Vec<&str> = vec!["good", "bad"];
    if include_skipped {
        statuses.push("skip");
    }
    if include_pending {
        statuses.push("pending");
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
    let mut stats = ExportStats::default();
    db::for_each_sample_export(conn, session_ids, &statuses, |row| {
        if human_only && row.decided_by.as_deref() == Some("classifier") {
            return Ok(());
        }
        let label: Option<i64> = match row.status.as_str() {
            "good" => Some(1),
            "bad" => Some(0),
            _ => None,
        };
        match row.status.as_str() {
            "good" => stats.good += 1,
            "bad" => stats.bad += 1,
            "skip" => stats.skip += 1,
            _ => stats.pending += 1,
        }
        match row.decided_by.as_deref() {
            Some("human") => stats.human += 1,
            Some("classifier") => stats.classifier += 1,
            _ => {}
        }
        let obj = serde_json::json!({
            "sample_id": row.sample_id,
            "session_id": row.session_id,
            "root": row.root_path,
            "path": row.rel_path,
            "format": row.format,
            "kind": row.kind,
            "compression": row.compression,
            "locator": row.locator,
            "content_hash": row.content_hash,
            "verdict": row.status,
            "label": label,
            "source": row.decided_by,
            "decided_at": row.decided_at,
            "tags": row.tags,
            "text": row.text,
            "text_bytes": row.text_bytes,
            "truncated": row.truncated,
        });
        let line = serde_json::to_vec(&obj)?;
        w.write_all(&line)?;
        w.write_all(b"\n")?;
        stats.rows += 1;
        stats.bytes += line.len() as u64 + 1;
        Ok(())
    })?;
    w.flush()?;
    Ok(stats)
}
