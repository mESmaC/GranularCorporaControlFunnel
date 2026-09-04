//! Session builder: picks N files from a root (seeded), cuts M samples from
//! each in parallel, and stores the samples with locators. Only the selected
//! files are ever opened.

use crate::db::{self, Db, FileRow, NewSample};
use crate::error::Result;
use crate::jobs::JobHandle;
use crate::rng::Rng;
use crate::sample::{content_hash, sample_file, SampleOutput, SampleParams};
use std::sync::Arc;

pub struct BuildConfig {
    pub threads: usize,
    pub max_bytes_per_sample: usize,
    pub max_stream_bytes: u64,
}

struct FileResult {
    file: FileRow,
    output: std::result::Result<SampleOutput, String>,
}

pub fn build_session(db: Arc<Db>, job: JobHandle, session_id: i64, cfg: BuildConfig) -> Result<()> {
    let session = {
        let conn = db.lock();
        db::get_session(&conn, session_id)?
    };

    // 1. Pick files.
    let files: Vec<FileRow> = {
        let conn = db.lock();
        let mut ids = db::eligible_file_ids(&conn, session.root_id, &session.format_filter, session.exclude_reviewed_files)?;
        let mut rng = Rng::derive(session.seed as u64, "file-selection");
        rng.shuffle(&mut ids);
        ids.truncate(session.files_requested.max(0) as usize);
        db::get_files(&conn, &ids)?
    };
    let total = files.len() as u64;
    if total == 0 {
        let conn = db.lock();
        db::finish_session(&conn, session_id, 0, "ready", Some("no eligible files matched"))?;
        job.finish("done", "no eligible files matched");
        return Ok(());
    }
    job.progress(0, total, format!("sampling {total} files"));

    let params = SampleParams {
        samples_per_file: session.samples_per_file.max(0) as usize,
        lines_per_sample: session.lines_per_sample.max(1) as usize,
        seed: session.seed as u64,
        max_bytes_per_sample: cfg.max_bytes_per_sample,
        max_stream_bytes: cfg.max_stream_bytes,
    };

    // 2. Sample in parallel; a dedicated thread drives the pool and streams
    //    results back so DB writes stay on this thread.
    let (tx, rx) = crossbeam_channel::bounded::<FileResult>(64);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.threads.max(1))
        .thread_name(|i| format!("gccf-sample-{i}"))
        .build()
        .map_err(|e| crate::error::GccfError::msg(e.to_string()))?;
    let job_for_pool = job.clone();
    let params_for_pool = params.clone();
    let driver = std::thread::spawn(move || {
        use rayon::prelude::*;
        pool.install(|| {
            files.into_par_iter().for_each_with(tx, |tx, file| {
                if job_for_pool.is_cancelled() {
                    return;
                }
                let path = file.abs_path();
                let output = sample_file(&path, file.format_enum(), file.kind_enum(), file.compression_enum(), &params_for_pool)
                    .map_err(|e| e.to_string());
                let _ = tx.send(FileResult { file, output });
            });
        });
    });

    let mut done: u64 = 0;
    let mut files_with_samples: i64 = 0;
    let mut samples_total: i64 = 0;
    let mut errors: Vec<String> = Vec::new();
    let mut next_ordinal: i64 = 0;

    for res in rx.iter() {
        done += 1;
        match res.output {
            Ok(out) => {
                if !out.samples.is_empty() {
                    files_with_samples += 1;
                }
                let hashes: Vec<String> = out.samples.iter().map(|s| content_hash(&s.text)).collect();
                let locs: Vec<String> = out
                    .samples
                    .iter()
                    .map(|s| serde_json::to_string(&s.locator).unwrap_or_else(|_| "{}".into()))
                    .collect();
                let batch: Vec<NewSample<'_>> = out
                    .samples
                    .iter()
                    .enumerate()
                    .map(|(i, s)| NewSample {
                        file_id: res.file.id,
                        locator_kind: s.locator.kind(),
                        locator_json: &locs[i],
                        text: &s.text,
                        truncated: s.truncated,
                        content_hash: &hashes[i],
                    })
                    .collect();
                let mut conn = db.lock();
                if !batch.is_empty() {
                    next_ordinal = db::insert_samples(&mut conn, session_id, next_ordinal, &batch)?;
                    samples_total += batch.len() as i64;
                }
                if let Some(rc) = out.row_count {
                    db::set_file_row_count(&conn, res.file.id, rc)?;
                }
                if let (Some(f), Some(k)) = (out.effective_format, out.effective_kind) {
                    if f.as_str() != res.file.format || k.as_str() != res.file.kind {
                        conn.execute(
                            "UPDATE files SET format = ?2, kind = ?3 WHERE id = ?1",
                            rusqlite::params![res.file.id, f.as_str(), k.as_str()],
                        )?;
                    }
                }
                if let Some(note) = out.note {
                    log::info!("{}: {note}", res.file.rel_path);
                }
            }
            Err(e) => {
                if errors.len() < 50 {
                    errors.push(format!("{}: {e}", res.file.rel_path));
                }
                log::warn!("sampling {}: {e}", res.file.rel_path);
            }
        }
        job.progress(done, total, format!("{done}/{total} files, {samples_total} samples"));
        if job.is_cancelled() {
            break;
        }
    }
    let _ = driver.join();

    // 3. Interleave files during review and finalise.
    {
        let mut conn = db.lock();
        db::shuffle_session_ordinals(&mut conn, session_id, session.seed as u64)?;
        let status = "ready";
        let err = if errors.is_empty() {
            None
        } else {
            Some(format!("{} file(s) failed: {}", errors.len(), errors.join("; ")))
        };
        db::finish_session(&conn, session_id, files_with_samples, status, err.as_deref())?;
    }
    let msg = if job.is_cancelled() {
        format!("cancelled after {done}/{total} files, {samples_total} samples kept")
    } else {
        format!("{samples_total} samples from {files_with_samples} files")
    };
    job.finish_with_payload(
        if job.is_cancelled() { "cancelled" } else { "done" },
        msg,
        serde_json::json!({ "session_id": session_id, "samples": samples_total, "files": files_with_samples, "errors": errors }),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn builds_session_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        for f in 0..6 {
            let mut fh = std::fs::File::create(dir.path().join(format!("shard{f}.jsonl"))).unwrap();
            for i in 0..300 {
                writeln!(fh, "{{\"id\": {i}, \"shard\": {f}}}").unwrap();
            }
        }
        let mut t = std::fs::File::create(dir.path().join("prose.txt")).unwrap();
        for i in 0..500 {
            writeln!(t, "sentence number {i} of the prose file").unwrap();
        }
        drop(t);

        let db = Arc::new(Db::open_memory().unwrap());
        let jobs = Arc::new(crate::jobs::Jobs::new());
        let job = jobs.start(None, "scan", 0, "", serde_json::Value::Null);
        let root_id = db::insert_root(&db.lock(), &dir.path().to_string_lossy(), None).unwrap();
        crate::scan::scan_root(&db, &job, root_id, dir.path()).unwrap();

        let sid = db::insert_session(
            &db.lock(),
            &db::NewSession {
                root_id,
                name: "t",
                files_requested: 4,
                samples_per_file: 10,
                lines_per_sample: 25,
                seed: 99,
                format_filter: &[],
                exclude_reviewed_files: true,
            },
        )
        .unwrap();
        let job = jobs.start(None, "session", 0, "", serde_json::Value::Null);
        build_session(
            db.clone(),
            job,
            sid,
            BuildConfig {
                threads: 3,
                max_bytes_per_sample: 65536,
                max_stream_bytes: 0,
            },
        )
        .unwrap();
        let s = db::get_session(&db.lock(), sid).unwrap();
        assert_eq!(s.status, "ready");
        assert_eq!(s.files_selected, 4);
        assert_eq!(s.samples_total, 40);
        assert_eq!(s.counts.pending, 40);

        // Same seed => same selection.
        let sid2 = db::insert_session(
            &db.lock(),
            &db::NewSession {
                root_id,
                name: "t2",
                files_requested: 4,
                samples_per_file: 10,
                lines_per_sample: 25,
                seed: 99,
                format_filter: &[],
                exclude_reviewed_files: false,
            },
        )
        .unwrap();
        let job = jobs.start(None, "session", 0, "", serde_json::Value::Null);
        build_session(db.clone(), job, sid2, BuildConfig { threads: 2, max_bytes_per_sample: 65536, max_stream_bytes: 0 }).unwrap();
        let conn = db.lock();
        let a: Vec<String> = {
            let mut st = conn.prepare("SELECT content_hash FROM samples WHERE session_id = ?1 ORDER BY content_hash").unwrap();
            st.query_map([sid], |r| r.get(0)).unwrap().collect::<std::result::Result<_, _>>().unwrap()
        };
        let b: Vec<String> = {
            let mut st = conn.prepare("SELECT content_hash FROM samples WHERE session_id = ?1 ORDER BY content_hash").unwrap();
            st.query_map([sid2], |r| r.get(0)).unwrap().collect::<std::result::Result<_, _>>().unwrap()
        };
        assert_eq!(a, b);
    }
}
