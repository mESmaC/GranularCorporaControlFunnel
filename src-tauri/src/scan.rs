//! Directory scan: builds the file index for a root without opening files
//! (except a 2 KiB sniff for extensionless names). On Windows the size and
//! mtime come from the directory listing itself, so this is a pure metadata
//! walk and handles millions of entries.

use crate::db::{self, Db, ScannedFile};
use crate::error::Result;
use crate::formats::{classify_name, sniff_is_text, Compression, Format, Kind};
use crate::jobs::JobHandle;
use std::path::Path;
use std::time::UNIX_EPOCH;
use walkdir::WalkDir;

const BATCH: usize = 2000;

pub fn scan_root(db: &Db, job: &JobHandle, root_id: i64, root_path: &Path) -> Result<(i64, i64)> {
    let generation = {
        let conn = db.lock();
        db::begin_scan_generation(&conn, root_id)?
    };

    let mut batch: Vec<ScannedFile> = Vec::with_capacity(BATCH);
    let mut seen: u64 = 0;
    let mut bytes: u64 = 0;
    let mut cancelled = false;

    let walker = WalkDir::new(root_path)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| e.file_name() != ".git");

    for entry in walker {
        if job.is_cancelled() {
            cancelled = true;
            break;
        }
        let entry = match entry {
            Ok(e) => e,
            Err(err) => {
                log::warn!("scan: {err}");
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        let rel = match entry.path().strip_prefix(root_path) {
            Ok(r) => r.to_string_lossy().to_string(),
            Err(_) => continue,
        };
        let name = entry.file_name().to_string_lossy();
        let (mut cls, ext) = classify_name(&name);
        if cls.format == Format::Unknown && ext.is_empty() && cls.compression == Compression::None {
            // Extensionless: cheap sniff so binaries don't pollute sampling.
            match sniff_is_text(entry.path()) {
                Some(true) => {
                    cls.format = Format::Text;
                    cls.kind = Kind::Lines;
                }
                _ => {
                    cls.format = Format::Binary;
                    cls.kind = Kind::Unsupported;
                }
            }
        }
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let size = meta.len();
        bytes += size;
        seen += 1;
        batch.push(ScannedFile {
            rel_path: rel,
            size_bytes: size as i64,
            mtime,
            ext,
            format: cls.format,
            kind: cls.kind,
            compression: cls.compression,
        });
        if batch.len() >= BATCH {
            let mut conn = db.lock();
            db::upsert_files(&mut conn, root_id, generation, &batch)?;
            batch.clear();
            job.progress(seen, 0, format!("{seen} files, {} GB", bytes as f64 / 1e9));
        }
    }
    if !batch.is_empty() {
        let mut conn = db.lock();
        db::upsert_files(&mut conn, root_id, generation, &batch)?;
    }

    let conn = db.lock();
    if cancelled {
        db::set_root_scan_status(&conn, root_id, "partial", Some("scan cancelled"))?;
        return Ok((seen as i64, bytes as i64));
    }
    let (count, total) = db::finish_scan(&conn, root_id, generation)?;
    Ok((count, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn scans_and_rescans() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("a.jsonl"), "{\"a\":1}\n").unwrap();
        std::fs::write(dir.path().join("sub/b.txt"), "hello\n").unwrap();
        std::fs::write(dir.path().join("sub/c.png"), [0u8, 1, 2]).unwrap();
        std::fs::write(dir.path().join("README"), "plain text readme").unwrap();
        std::fs::write(dir.path().join("blob"), [0u8, 0, 0, 1]).unwrap();

        let db = Db::open_memory().unwrap();
        let jobs = Arc::new(crate::jobs::Jobs::new());
        let job = jobs.start(None, "scan", 0, "", serde_json::Value::Null);
        let root_id = db::insert_root(&db.lock(), &dir.path().to_string_lossy(), None).unwrap();
        let (count, _) = scan_root(&db, &job, root_id, dir.path()).unwrap();
        assert_eq!(count, 5);
        let stats = db::root_format_stats(&db.lock(), root_id).unwrap();
        let binaries: i64 = stats.iter().filter(|s| s.format == "binary").map(|s| s.count).sum();
        assert_eq!(binaries, 2);
        let eligible = db::eligible_file_ids(&db.lock(), root_id, &[], false).unwrap();
        assert_eq!(eligible.len(), 3);

        // Remove a file and rescan: it should be flagged missing, not deleted.
        std::fs::remove_file(dir.path().join("sub/b.txt")).unwrap();
        let (count, _) = scan_root(&db, &job, root_id, dir.path()).unwrap();
        assert_eq!(count, 4);
        let missing: i64 = db
            .lock()
            .query_row("SELECT COUNT(*) FROM files WHERE missing = 1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(missing, 1);
    }
}
