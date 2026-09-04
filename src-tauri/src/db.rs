//! SQLite catalogue. This is the *only* durable state GCCF keeps: nothing in the
//! corpora is ever copied. Files are indexed, samples are cut and stored as
//! small text snippets with a locator back into the source file, and every
//! human / classifier decision is appended to an event log.

use crate::error::{GccfError, Result};
use crate::formats::{Compression, Format, Kind};
use rusqlite::{params, params_from_iter, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const SCHEMA_VERSION: i64 = 2;

pub const BUILTIN_TAGS: &[&str] = &[
    "code",
    "english",
    "non-english",
    "academic",
    "agent-trace",
    "conversational",
    "math",
    "web",
    "synthetic",
    "boilerplate",
    "pii",
];

pub struct Db {
    pub conn: Mutex<Connection>,
    pub path: PathBuf,
}

pub fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

impl Db {
    pub fn open(path: &Path) -> Result<Db> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA foreign_keys = ON;
             PRAGMA temp_store = MEMORY;
             PRAGMA cache_size = -65536;",
        )?;
        let db = Db {
            conn: Mutex::new(conn),
            path: path.to_path_buf(),
        };
        db.migrate()?;
        Ok(db)
    }

    #[cfg(test)]
    pub fn open_memory() -> Result<Db> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        let db = Db {
            conn: Mutex::new(conn),
            path: PathBuf::from(":memory:"),
        };
        db.migrate()?;
        Ok(db)
    }

    pub fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        match self.conn.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn migrate(&self) -> Result<()> {
        let conn = self.lock();
        conn.execute_batch(SCHEMA)?;
        ensure_column(&conn, "classifier_runs", "model_row_id", "INTEGER")?;
        ensure_column(&conn, "classifier_runs", "mode", "TEXT NOT NULL DEFAULT 'score'")?;
        let v: i64 = conn
            .query_row("SELECT value FROM settings WHERE key = 'schema_version'", [], |r| r.get::<_, String>(0))
            .optional()?
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if v < SCHEMA_VERSION {
            conn.execute(
                "INSERT OR REPLACE INTO settings(key, value) VALUES('schema_version', ?1)",
                params![SCHEMA_VERSION.to_string()],
            )?;
        }
        let now = now_iso();
        for (i, name) in BUILTIN_TAGS.iter().enumerate() {
            let hotkey: Option<i64> = if i < 9 { Some(i as i64 + 1) } else { None };
            conn.execute(
                "INSERT OR IGNORE INTO tags(name, builtin, hotkey, created_at) VALUES(?1, 1, ?2, ?3)",
                params![name, hotkey, now],
            )?;
        }
        Ok(())
    }
}

/// Add a column to an existing table if it is missing (SQLite has no
/// `ADD COLUMN IF NOT EXISTS`).
fn ensure_column(conn: &Connection, table: &str, column: &str, decl: &str) -> Result<()> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let exists = stmt
        .query_map([], |r| r.get::<_, String>(1))?
        .filter_map(|r| r.ok())
        .any(|c| c == column);
    if !exists {
        conn.execute(&format!("ALTER TABLE {table} ADD COLUMN {column} {decl}"), [])?;
    }
    Ok(())
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS roots (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  label TEXT,
  created_at TEXT NOT NULL,
  scanned_at TEXT,
  scan_generation INTEGER NOT NULL DEFAULT 0,
  file_count INTEGER NOT NULL DEFAULT 0,
  total_bytes INTEGER NOT NULL DEFAULT 0,
  scan_status TEXT NOT NULL DEFAULT 'pending',
  scan_error TEXT
);

CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY,
  root_id INTEGER NOT NULL REFERENCES roots(id) ON DELETE CASCADE,
  rel_path TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  mtime INTEGER NOT NULL,
  ext TEXT NOT NULL,
  format TEXT NOT NULL,
  kind TEXT NOT NULL,
  compression TEXT NOT NULL,
  row_count INTEGER,
  last_seen_scan INTEGER NOT NULL,
  missing INTEGER NOT NULL DEFAULT 0,
  UNIQUE(root_id, rel_path)
);
CREATE INDEX IF NOT EXISTS files_root_format ON files(root_id, format, kind);
CREATE INDEX IF NOT EXISTS files_root_missing ON files(root_id, missing);

CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY,
  root_id INTEGER NOT NULL REFERENCES roots(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL,
  files_requested INTEGER NOT NULL,
  samples_per_file INTEGER NOT NULL,
  lines_per_sample INTEGER NOT NULL,
  seed INTEGER NOT NULL,
  format_filter TEXT,
  exclude_reviewed_files INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL DEFAULT 'building',
  error TEXT,
  files_selected INTEGER NOT NULL DEFAULT 0,
  samples_total INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS samples (
  id INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  file_id INTEGER NOT NULL REFERENCES files(id),
  ordinal INTEGER NOT NULL,
  locator_kind TEXT NOT NULL,
  locator_json TEXT NOT NULL,
  text TEXT NOT NULL,
  text_bytes INTEGER NOT NULL,
  truncated INTEGER NOT NULL DEFAULT 0,
  content_hash TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  decided_at TEXT,
  decided_by TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS samples_session_status ON samples(session_id, status, ordinal);
CREATE INDEX IF NOT EXISTS samples_file ON samples(file_id);
CREATE INDEX IF NOT EXISTS samples_hash ON samples(content_hash);

CREATE TABLE IF NOT EXISTS annotations (
  id INTEGER PRIMARY KEY,
  sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
  verdict TEXT NOT NULL,
  tags_json TEXT NOT NULL,
  source TEXT NOT NULL,
  model_id TEXT,
  score REAL,
  view_ms INTEGER,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS annotations_sample ON annotations(sample_id, created_at);

CREATE TABLE IF NOT EXISTS tags (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  builtin INTEGER NOT NULL DEFAULT 0,
  hotkey INTEGER,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sample_tags (
  sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
  tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
  source TEXT NOT NULL DEFAULT 'human',
  PRIMARY KEY(sample_id, tag_id)
);
CREATE INDEX IF NOT EXISTS sample_tags_tag ON sample_tags(tag_id);

CREATE TABLE IF NOT EXISTS classifier_runs (
  id INTEGER PRIMARY KEY,
  session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
  backend TEXT NOT NULL,
  model_id TEXT NOT NULL,
  command TEXT,
  workers INTEGER NOT NULL,
  batch_size INTEGER NOT NULL,
  status TEXT NOT NULL,
  total INTEGER NOT NULL DEFAULT 0,
  completed INTEGER NOT NULL DEFAULT 0,
  failed INTEGER NOT NULL DEFAULT 0,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  error TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY,
  sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
  run_id INTEGER NOT NULL REFERENCES classifier_runs(id) ON DELETE CASCADE,
  verdict TEXT NOT NULL,
  score REAL NOT NULL,
  tags_json TEXT NOT NULL,
  latency_ms INTEGER,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS predictions_sample ON predictions(sample_id, id);
CREATE INDEX IF NOT EXISTS predictions_run ON predictions(run_id);

CREATE TABLE IF NOT EXISTS models (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  command TEXT,
  parent_id INTEGER REFERENCES models(id) ON DELETE SET NULL,
  created_at TEXT NOT NULL,
  train_rows INTEGER NOT NULL DEFAULT 0,
  metrics_json TEXT,
  tag_names_json TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS eval_feedback (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES classifier_runs(id) ON DELETE CASCADE,
  sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
  predicted TEXT NOT NULL,
  score REAL NOT NULL,
  predicted_tags_json TEXT NOT NULL,
  human TEXT NOT NULL,
  human_tags_json TEXT NOT NULL,
  blind INTEGER NOT NULL DEFAULT 1,
  view_ms INTEGER,
  created_at TEXT NOT NULL,
  UNIQUE(run_id, sample_id)
);
CREATE INDEX IF NOT EXISTS eval_feedback_sample ON eval_feedback(sample_id);
"#;

// ---------------------------------------------------------------------------
// Data transfer types (serialized to the frontend)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootInfo {
    pub id: i64,
    pub path: String,
    pub label: Option<String>,
    pub created_at: String,
    pub scanned_at: Option<String>,
    pub file_count: i64,
    pub total_bytes: i64,
    pub scan_status: String,
    pub scan_error: Option<String>,
    pub sampleable_files: i64,
    pub sampleable_bytes: i64,
    pub sessions: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatStat {
    pub format: String,
    pub kind: String,
    pub compression: String,
    pub count: i64,
    pub bytes: i64,
    pub sampled_files: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StatusCounts {
    pub pending: i64,
    pub good: i64,
    pub bad: i64,
    pub skip: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub id: i64,
    pub root_id: i64,
    pub root_path: String,
    pub name: String,
    pub created_at: String,
    pub files_requested: i64,
    pub samples_per_file: i64,
    pub lines_per_sample: i64,
    pub seed: i64,
    pub format_filter: Vec<String>,
    pub exclude_reviewed_files: bool,
    pub status: String,
    pub error: Option<String>,
    pub files_selected: i64,
    pub samples_total: i64,
    pub counts: StatusCounts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRow {
    pub id: i64,
    pub root_id: i64,
    pub root_path: String,
    pub rel_path: String,
    pub size_bytes: i64,
    pub mtime: i64,
    pub ext: String,
    pub format: String,
    pub kind: String,
    pub compression: String,
    pub row_count: Option<i64>,
}

impl FileRow {
    pub fn abs_path(&self) -> PathBuf {
        Path::new(&self.root_path).join(&self.rel_path)
    }
    pub fn format_enum(&self) -> Format {
        Format::parse(&self.format)
    }
    pub fn kind_enum(&self) -> Kind {
        Kind::parse(&self.kind)
    }
    pub fn compression_enum(&self) -> Compression {
        Compression::parse(&self.compression)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionView {
    pub run_id: i64,
    pub model_id: String,
    pub verdict: String,
    pub score: f64,
    pub tags: Vec<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleView {
    pub id: i64,
    pub session_id: i64,
    pub file_id: i64,
    pub ordinal: i64,
    pub rel_path: String,
    pub abs_path: String,
    pub format: String,
    pub kind: String,
    pub compression: String,
    pub file_size: i64,
    pub locator_kind: String,
    pub locator: serde_json::Value,
    pub text: String,
    pub text_bytes: i64,
    pub truncated: bool,
    pub content_hash: String,
    pub status: String,
    pub decided_at: Option<String>,
    pub decided_by: Option<String>,
    pub tags: Vec<i64>,
    pub prediction: Option<PredictionView>,
    pub duplicates: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleSummary {
    pub id: i64,
    pub session_id: i64,
    pub session_name: String,
    pub file_id: i64,
    pub rel_path: String,
    pub format: String,
    pub ordinal: i64,
    pub status: String,
    pub preview: String,
    pub decided_at: Option<String>,
    pub tags: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagInfo {
    pub id: i64,
    pub name: String,
    pub builtin: bool,
    pub hotkey: Option<i64>,
    pub usage: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierRunInfo {
    pub id: i64,
    pub mode: String,
    pub model_row_id: Option<i64>,
    pub session_id: Option<i64>,
    pub session_name: Option<String>,
    pub backend: String,
    pub model_id: String,
    pub command: Option<String>,
    pub workers: i64,
    pub batch_size: i64,
    pub status: String,
    pub total: i64,
    pub completed: i64,
    pub failed: i64,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub error: Option<String>,
    pub agreement: Option<AgreementStats>,
    pub evaluated: i64,
    pub eval_agree: i64,
}

/// How predictions line up with human decisions on the same samples.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgreementStats {
    pub compared: i64,
    pub agree: i64,
    pub tp: i64,
    pub fp: i64,
    pub tn: i64,
    pub fn_: i64,
}

// ---------------------------------------------------------------------------
// Roots & files
// ---------------------------------------------------------------------------

pub fn insert_root(conn: &Connection, path: &str, label: Option<&str>) -> Result<i64> {
    conn.execute(
        "INSERT INTO roots(path, label, created_at) VALUES(?1, ?2, ?3)",
        params![path, label, now_iso()],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn get_root_path(conn: &Connection, root_id: i64) -> Result<String> {
    conn.query_row("SELECT path FROM roots WHERE id = ?1", params![root_id], |r| r.get(0))
        .optional()?
        .ok_or_else(|| GccfError::msg(format!("root {root_id} not found")))
}

pub fn list_roots(conn: &Connection) -> Result<Vec<RootInfo>> {
    let mut stmt = conn.prepare(
        "SELECT r.id, r.path, r.label, r.created_at, r.scanned_at, r.file_count, r.total_bytes, r.scan_status, r.scan_error,
                (SELECT COUNT(*) FROM files f WHERE f.root_id = r.id AND f.missing = 0 AND f.kind != 'unsupported' AND f.size_bytes > 0),
                (SELECT COALESCE(SUM(size_bytes),0) FROM files f WHERE f.root_id = r.id AND f.missing = 0 AND f.kind != 'unsupported'),
                (SELECT COUNT(*) FROM sessions s WHERE s.root_id = r.id)
         FROM roots r ORDER BY r.id",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok(RootInfo {
            id: r.get(0)?,
            path: r.get(1)?,
            label: r.get(2)?,
            created_at: r.get(3)?,
            scanned_at: r.get(4)?,
            file_count: r.get(5)?,
            total_bytes: r.get(6)?,
            scan_status: r.get(7)?,
            scan_error: r.get(8)?,
            sampleable_files: r.get(9)?,
            sampleable_bytes: r.get(10)?,
            sessions: r.get(11)?,
        })
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn delete_root(conn: &Connection, root_id: i64) -> Result<()> {
    conn.execute("DELETE FROM roots WHERE id = ?1", params![root_id])?;
    Ok(())
}

pub fn set_root_scan_status(conn: &Connection, root_id: i64, status: &str, error: Option<&str>) -> Result<()> {
    conn.execute(
        "UPDATE roots SET scan_status = ?2, scan_error = ?3 WHERE id = ?1",
        params![root_id, status, error],
    )?;
    Ok(())
}

pub fn begin_scan_generation(conn: &Connection, root_id: i64) -> Result<i64> {
    conn.execute(
        "UPDATE roots SET scan_generation = scan_generation + 1, scan_status = 'scanning', scan_error = NULL WHERE id = ?1",
        params![root_id],
    )?;
    Ok(conn.query_row("SELECT scan_generation FROM roots WHERE id = ?1", params![root_id], |r| r.get(0))?)
}

pub struct ScannedFile {
    pub rel_path: String,
    pub size_bytes: i64,
    pub mtime: i64,
    pub ext: String,
    pub format: Format,
    pub kind: Kind,
    pub compression: Compression,
}

/// Upsert a batch of scanned files inside one transaction.
pub fn upsert_files(conn: &mut Connection, root_id: i64, generation: i64, batch: &[ScannedFile]) -> Result<()> {
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare_cached(
            "INSERT INTO files(root_id, rel_path, size_bytes, mtime, ext, format, kind, compression, last_seen_scan, missing)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 0)
             ON CONFLICT(root_id, rel_path) DO UPDATE SET
               size_bytes = excluded.size_bytes,
               mtime = excluded.mtime,
               ext = excluded.ext,
               format = excluded.format,
               kind = excluded.kind,
               compression = excluded.compression,
               last_seen_scan = excluded.last_seen_scan,
               missing = 0,
               row_count = CASE WHEN files.size_bytes = excluded.size_bytes AND files.mtime = excluded.mtime THEN files.row_count ELSE NULL END",
        )?;
        for f in batch {
            stmt.execute(params![
                root_id,
                f.rel_path,
                f.size_bytes,
                f.mtime,
                f.ext,
                f.format.as_str(),
                f.kind.as_str(),
                f.compression.as_str(),
                generation
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}

pub fn finish_scan(conn: &Connection, root_id: i64, generation: i64) -> Result<(i64, i64)> {
    conn.execute(
        "UPDATE files SET missing = 1 WHERE root_id = ?1 AND last_seen_scan < ?2",
        params![root_id, generation],
    )?;
    let (count, bytes): (i64, i64) = conn.query_row(
        "SELECT COUNT(*), COALESCE(SUM(size_bytes),0) FROM files WHERE root_id = ?1 AND missing = 0",
        params![root_id],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    conn.execute(
        "UPDATE roots SET scanned_at = ?2, file_count = ?3, total_bytes = ?4, scan_status = 'done', scan_error = NULL WHERE id = ?1",
        params![root_id, now_iso(), count, bytes],
    )?;
    Ok((count, bytes))
}

pub fn root_format_stats(conn: &Connection, root_id: i64) -> Result<Vec<FormatStat>> {
    let mut stmt = conn.prepare(
        "SELECT f.format, f.kind, f.compression, COUNT(*), COALESCE(SUM(f.size_bytes),0),
                SUM(CASE WHEN EXISTS(SELECT 1 FROM samples s WHERE s.file_id = f.id) THEN 1 ELSE 0 END)
         FROM files f WHERE f.root_id = ?1 AND f.missing = 0
         GROUP BY f.format, f.kind, f.compression
         ORDER BY COUNT(*) DESC",
    )?;
    let rows = stmt.query_map(params![root_id], |r| {
        Ok(FormatStat {
            format: r.get(0)?,
            kind: r.get(1)?,
            compression: r.get(2)?,
            count: r.get(3)?,
            bytes: r.get(4)?,
            sampled_files: r.get(5)?,
        })
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

fn file_row_from(r: &Row<'_>) -> rusqlite::Result<FileRow> {
    Ok(FileRow {
        id: r.get(0)?,
        root_id: r.get(1)?,
        root_path: r.get(2)?,
        rel_path: r.get(3)?,
        size_bytes: r.get(4)?,
        mtime: r.get(5)?,
        ext: r.get(6)?,
        format: r.get(7)?,
        kind: r.get(8)?,
        compression: r.get(9)?,
        row_count: r.get(10)?,
    })
}

const FILE_SELECT: &str = "SELECT f.id, f.root_id, r.path, f.rel_path, f.size_bytes, f.mtime, f.ext, f.format, f.kind, f.compression, f.row_count
                           FROM files f JOIN roots r ON r.id = f.root_id";

pub fn get_file(conn: &Connection, file_id: i64) -> Result<FileRow> {
    conn.query_row(&format!("{FILE_SELECT} WHERE f.id = ?1"), params![file_id], file_row_from)
        .optional()?
        .ok_or_else(|| GccfError::msg(format!("file {file_id} not found")))
}

pub fn get_files(conn: &Connection, ids: &[i64]) -> Result<Vec<FileRow>> {
    let mut out = Vec::with_capacity(ids.len());
    let mut stmt = conn.prepare_cached(&format!("{FILE_SELECT} WHERE f.id = ?1"))?;
    for id in ids {
        if let Some(f) = stmt.query_row(params![id], file_row_from).optional()? {
            out.push(f);
        }
    }
    Ok(out)
}

/// IDs of files eligible for sampling in a root.
pub fn eligible_file_ids(
    conn: &Connection,
    root_id: i64,
    formats: &[String],
    exclude_reviewed: bool,
) -> Result<Vec<i64>> {
    let mut sql = String::from(
        "SELECT f.id FROM files f WHERE f.root_id = ?1 AND f.missing = 0 AND f.kind != 'unsupported' AND f.size_bytes > 0",
    );
    let mut args: Vec<rusqlite::types::Value> = vec![root_id.into()];
    if !formats.is_empty() {
        sql.push_str(" AND f.format IN (");
        for (i, f) in formats.iter().enumerate() {
            if i > 0 {
                sql.push(',');
            }
            sql.push('?');
            args.push(f.clone().into());
        }
        sql.push(')');
    }
    if exclude_reviewed {
        sql.push_str(" AND NOT EXISTS (SELECT 1 FROM samples s WHERE s.file_id = f.id)");
    }
    sql.push_str(" ORDER BY f.id");
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(args.iter()), |r| r.get::<_, i64>(0))?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn set_file_row_count(conn: &Connection, file_id: i64, row_count: i64) -> Result<()> {
    conn.execute("UPDATE files SET row_count = ?2 WHERE id = ?1", params![file_id, row_count])?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

pub struct NewSession<'a> {
    pub root_id: i64,
    pub name: &'a str,
    pub files_requested: i64,
    pub samples_per_file: i64,
    pub lines_per_sample: i64,
    pub seed: i64,
    pub format_filter: &'a [String],
    pub exclude_reviewed_files: bool,
}

pub fn insert_session(conn: &Connection, s: &NewSession<'_>) -> Result<i64> {
    let filter = if s.format_filter.is_empty() {
        None
    } else {
        Some(serde_json::to_string(s.format_filter)?)
    };
    conn.execute(
        "INSERT INTO sessions(root_id, name, created_at, files_requested, samples_per_file, lines_per_sample, seed, format_filter, exclude_reviewed_files, status)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'building')",
        params![
            s.root_id,
            s.name,
            now_iso(),
            s.files_requested,
            s.samples_per_file,
            s.lines_per_sample,
            s.seed,
            filter,
            s.exclude_reviewed_files as i64
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn finish_session(conn: &Connection, session_id: i64, files_selected: i64, status: &str, error: Option<&str>) -> Result<()> {
    let total: i64 = conn.query_row(
        "SELECT COUNT(*) FROM samples WHERE session_id = ?1",
        params![session_id],
        |r| r.get(0),
    )?;
    conn.execute(
        "UPDATE sessions SET files_selected = ?2, samples_total = ?3, status = ?4, error = ?5 WHERE id = ?1",
        params![session_id, files_selected, total, status, error],
    )?;
    Ok(())
}

fn session_from(r: &Row<'_>) -> rusqlite::Result<SessionInfo> {
    let filter: Option<String> = r.get(9)?;
    let format_filter = filter
        .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
        .unwrap_or_default();
    Ok(SessionInfo {
        id: r.get(0)?,
        root_id: r.get(1)?,
        root_path: r.get(2)?,
        name: r.get(3)?,
        created_at: r.get(4)?,
        files_requested: r.get(5)?,
        samples_per_file: r.get(6)?,
        lines_per_sample: r.get(7)?,
        seed: r.get(8)?,
        format_filter,
        exclude_reviewed_files: r.get::<_, i64>(10)? != 0,
        status: r.get(11)?,
        error: r.get(12)?,
        files_selected: r.get(13)?,
        samples_total: r.get(14)?,
        counts: StatusCounts {
            pending: r.get(15)?,
            good: r.get(16)?,
            bad: r.get(17)?,
            skip: r.get(18)?,
        },
    })
}

const SESSION_SELECT: &str = "SELECT s.id, s.root_id, r.path, s.name, s.created_at, s.files_requested, s.samples_per_file, s.lines_per_sample, s.seed,
        s.format_filter, s.exclude_reviewed_files, s.status, s.error, s.files_selected, s.samples_total,
        (SELECT COUNT(*) FROM samples x WHERE x.session_id = s.id AND x.status = 'pending'),
        (SELECT COUNT(*) FROM samples x WHERE x.session_id = s.id AND x.status = 'good'),
        (SELECT COUNT(*) FROM samples x WHERE x.session_id = s.id AND x.status = 'bad'),
        (SELECT COUNT(*) FROM samples x WHERE x.session_id = s.id AND x.status = 'skip')
        FROM sessions s JOIN roots r ON r.id = s.root_id";

pub fn list_sessions(conn: &Connection) -> Result<Vec<SessionInfo>> {
    let mut stmt = conn.prepare(&format!("{SESSION_SELECT} ORDER BY s.id DESC"))?;
    let rows = stmt.query_map([], session_from)?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn get_session(conn: &Connection, id: i64) -> Result<SessionInfo> {
    conn.query_row(&format!("{SESSION_SELECT} WHERE s.id = ?1"), params![id], session_from)
        .optional()?
        .ok_or_else(|| GccfError::msg(format!("session {id} not found")))
}

pub fn delete_session(conn: &Connection, id: i64) -> Result<()> {
    conn.execute("DELETE FROM sessions WHERE id = ?1", params![id])?;
    Ok(())
}

pub fn rename_session(conn: &Connection, id: i64, name: &str) -> Result<()> {
    conn.execute("UPDATE sessions SET name = ?2 WHERE id = ?1", params![id, name])?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Samples
// ---------------------------------------------------------------------------

pub struct NewSample<'a> {
    pub file_id: i64,
    pub locator_kind: &'a str,
    pub locator_json: &'a str,
    pub text: &'a str,
    pub truncated: bool,
    pub content_hash: &'a str,
}

pub fn insert_samples(conn: &mut Connection, session_id: i64, start_ordinal: i64, batch: &[NewSample<'_>]) -> Result<i64> {
    let tx = conn.transaction()?;
    let mut ordinal = start_ordinal;
    {
        let mut stmt = tx.prepare_cached(
            "INSERT INTO samples(session_id, file_id, ordinal, locator_kind, locator_json, text, text_bytes, truncated, content_hash, status, created_at)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'pending', ?10)",
        )?;
        let now = now_iso();
        for s in batch {
            stmt.execute(params![
                session_id,
                s.file_id,
                ordinal,
                s.locator_kind,
                s.locator_json,
                s.text,
                s.text.len() as i64,
                s.truncated as i64,
                s.content_hash,
                now
            ])?;
            ordinal += 1;
        }
    }
    tx.commit()?;
    Ok(ordinal)
}

/// Shuffle sample ordinals within a session so review interleaves files.
pub fn shuffle_session_ordinals(conn: &mut Connection, session_id: i64, seed: u64) -> Result<()> {
    let ids: Vec<i64> = {
        let mut stmt = conn.prepare("SELECT id FROM samples WHERE session_id = ?1 ORDER BY id")?;
        let rows = stmt.query_map(params![session_id], |r| r.get(0))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };
    let mut order: Vec<i64> = ids.clone();
    let mut rng = crate::rng::Rng::derive(seed, "ordinals");
    rng.shuffle(&mut order);
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare_cached("UPDATE samples SET ordinal = ?2 WHERE id = ?1")?;
        for (i, id) in order.iter().enumerate() {
            stmt.execute(params![id, i as i64])?;
        }
    }
    tx.commit()?;
    Ok(())
}

fn sample_view_from(conn: &Connection, r: &Row<'_>) -> rusqlite::Result<SampleView> {
    let id: i64 = r.get(0)?;
    let locator_json: String = r.get(11)?;
    let content_hash: String = r.get(15)?;
    let locator = serde_json::from_str(&locator_json).unwrap_or(serde_json::Value::Null);
    let tags = sample_tag_ids(conn, id)?;
    let prediction = latest_prediction(conn, id)?;
    let duplicates: i64 = conn.query_row(
        "SELECT COUNT(*) FROM samples WHERE content_hash = ?1 AND id != ?2",
        params![content_hash, id],
        |r| r.get(0),
    )?;
    Ok(SampleView {
        id,
        session_id: r.get(1)?,
        file_id: r.get(2)?,
        ordinal: r.get(3)?,
        rel_path: r.get(4)?,
        abs_path: {
            let root: String = r.get(5)?;
            let rel: String = r.get(4)?;
            Path::new(&root).join(rel).to_string_lossy().to_string()
        },
        format: r.get(6)?,
        kind: r.get(7)?,
        compression: r.get(8)?,
        file_size: r.get(9)?,
        locator_kind: r.get(10)?,
        locator,
        text: r.get(12)?,
        text_bytes: r.get(13)?,
        truncated: r.get::<_, i64>(14)? != 0,
        content_hash,
        status: r.get(16)?,
        decided_at: r.get(17)?,
        decided_by: r.get(18)?,
        tags,
        prediction,
        duplicates,
    })
}

const SAMPLE_SELECT: &str = "SELECT s.id, s.session_id, s.file_id, s.ordinal, f.rel_path, r.path, f.format, f.kind, f.compression, f.size_bytes,
        s.locator_kind, s.locator_json, s.text, s.text_bytes, s.truncated, s.content_hash, s.status, s.decided_at, s.decided_by
        FROM samples s JOIN files f ON f.id = s.file_id JOIN roots r ON r.id = f.root_id";

pub fn get_sample(conn: &Connection, id: i64) -> Result<SampleView> {
    let mut stmt = conn.prepare_cached(&format!("{SAMPLE_SELECT} WHERE s.id = ?1"))?;
    stmt.query_row(params![id], |r| sample_view_from(conn, r))
        .optional()?
        .ok_or_else(|| GccfError::msg(format!("sample {id} not found")))
}

/// Next sample to review. `statuses` restricts to e.g. ["pending"] or ["skip"].
/// `after_ordinal` continues past a given position (wraps to the start if
/// nothing is left after it).
pub fn next_sample(conn: &Connection, session_id: i64, statuses: &[&str], after_ordinal: Option<i64>) -> Result<Option<SampleView>> {
    let placeholders = statuses.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let base = format!("{SAMPLE_SELECT} WHERE s.session_id = ?1 AND s.status IN ({placeholders})");
    let mut args: Vec<rusqlite::types::Value> = vec![session_id.into()];
    for s in statuses {
        args.push((*s).to_string().into());
    }
    if let Some(after) = after_ordinal {
        let sql = format!("{base} AND s.ordinal > ? ORDER BY s.ordinal LIMIT 1");
        let mut a = args.clone();
        a.push(after.into());
        let mut stmt = conn.prepare_cached(&sql)?;
        if let Some(v) = stmt
            .query_row(params_from_iter(a.iter()), |r| sample_view_from(conn, r))
            .optional()?
        {
            return Ok(Some(v));
        }
    }
    let sql = format!("{base} ORDER BY s.ordinal LIMIT 1");
    let mut stmt = conn.prepare_cached(&sql)?;
    Ok(stmt
        .query_row(params_from_iter(args.iter()), |r| sample_view_from(conn, r))
        .optional()?)
}

pub fn session_counts(conn: &Connection, session_id: i64) -> Result<StatusCounts> {
    let mut stmt = conn.prepare_cached("SELECT status, COUNT(*) FROM samples WHERE session_id = ?1 GROUP BY status")?;
    let mut counts = StatusCounts::default();
    let rows = stmt.query_map(params![session_id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
    for row in rows {
        let (status, n) = row?;
        match status.as_str() {
            "pending" => counts.pending = n,
            "good" => counts.good = n,
            "bad" => counts.bad = n,
            "skip" => counts.skip = n,
            _ => {}
        }
    }
    Ok(counts)
}

pub fn list_samples(
    conn: &Connection,
    session_id: Option<i64>,
    statuses: &[&str],
    limit: i64,
    offset: i64,
) -> Result<Vec<SampleSummary>> {
    let placeholders = statuses.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let mut sql = format!(
        "SELECT s.id, s.session_id, se.name, s.file_id, f.rel_path, f.format, s.ordinal, s.status, substr(s.text, 1, 240), s.decided_at
         FROM samples s JOIN files f ON f.id = s.file_id JOIN sessions se ON se.id = s.session_id
         WHERE s.status IN ({placeholders})"
    );
    let mut args: Vec<rusqlite::types::Value> = statuses.iter().map(|s| (*s).to_string().into()).collect();
    if let Some(sid) = session_id {
        sql.push_str(" AND s.session_id = ?");
        args.push(sid.into());
    }
    sql.push_str(" ORDER BY s.decided_at DESC, s.ordinal LIMIT ? OFFSET ?");
    args.push(limit.into());
    args.push(offset.into());
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(args.iter()), |r| {
        Ok(SampleSummary {
            id: r.get(0)?,
            session_id: r.get(1)?,
            session_name: r.get(2)?,
            file_id: r.get(3)?,
            rel_path: r.get(4)?,
            format: r.get(5)?,
            ordinal: r.get(6)?,
            status: r.get(7)?,
            preview: r.get(8)?,
            decided_at: r.get(9)?,
            tags: Vec::new(),
        })
    })?;
    let mut out = rows.collect::<std::result::Result<Vec<_>, _>>()?;
    for s in &mut out {
        s.tags = sample_tag_ids(conn, s.id)?;
    }
    Ok(out)
}

/// Record a verdict for a sample. Appends an annotation event and updates the
/// denormalised status on the sample row.
pub fn decide(
    conn: &Connection,
    sample_id: i64,
    verdict: &str,
    source: &str,
    model_id: Option<&str>,
    score: Option<f64>,
    view_ms: Option<i64>,
) -> Result<()> {
    if !matches!(verdict, "good" | "bad" | "skip" | "pending") {
        return Err(GccfError::msg(format!("invalid verdict {verdict}")));
    }
    let tags = sample_tag_names(conn, sample_id)?;
    let tags_json = serde_json::to_string(&tags)?;
    let now = now_iso();
    conn.execute(
        "INSERT INTO annotations(sample_id, verdict, tags_json, source, model_id, score, view_ms, created_at)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![sample_id, verdict, tags_json, source, model_id, score, view_ms, now],
    )?;
    if verdict == "pending" {
        conn.execute(
            "UPDATE samples SET status = 'pending', decided_at = NULL, decided_by = NULL WHERE id = ?1",
            params![sample_id],
        )?;
    } else {
        conn.execute(
            "UPDATE samples SET status = ?2, decided_at = ?3, decided_by = ?4 WHERE id = ?1",
            params![sample_id, verdict, now, source],
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tags
// ---------------------------------------------------------------------------

pub fn list_tags(conn: &Connection) -> Result<Vec<TagInfo>> {
    let mut stmt = conn.prepare(
        "SELECT t.id, t.name, t.builtin, t.hotkey, (SELECT COUNT(*) FROM sample_tags st WHERE st.tag_id = t.id)
         FROM tags t ORDER BY CASE WHEN t.hotkey IS NULL THEN 1 ELSE 0 END, t.hotkey, t.id",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok(TagInfo {
            id: r.get(0)?,
            name: r.get(1)?,
            builtin: r.get::<_, i64>(2)? != 0,
            hotkey: r.get(3)?,
            usage: r.get(4)?,
        })
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn normalize_tag(name: &str) -> String {
    name.trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' || c == '+' || c == '#' { c } else { '-' })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

pub fn create_tag(conn: &Connection, name: &str) -> Result<i64> {
    let name = normalize_tag(name);
    if name.is_empty() {
        return Err(GccfError::msg("tag name is empty"));
    }
    conn.execute(
        "INSERT OR IGNORE INTO tags(name, builtin, created_at) VALUES(?1, 0, ?2)",
        params![name, now_iso()],
    )?;
    Ok(conn.query_row("SELECT id FROM tags WHERE name = ?1", params![name], |r| r.get(0))?)
}

pub fn get_or_create_tag_ids(conn: &Connection, names: &[String]) -> Result<Vec<i64>> {
    let mut out = Vec::with_capacity(names.len());
    for n in names {
        let norm = normalize_tag(n);
        if norm.is_empty() {
            continue;
        }
        out.push(create_tag(conn, &norm)?);
    }
    Ok(out)
}

pub fn delete_tag(conn: &Connection, id: i64) -> Result<()> {
    conn.execute("DELETE FROM tags WHERE id = ?1", params![id])?;
    Ok(())
}

pub fn rename_tag(conn: &Connection, id: i64, name: &str) -> Result<()> {
    let name = normalize_tag(name);
    if name.is_empty() {
        return Err(GccfError::msg("tag name is empty"));
    }
    conn.execute("UPDATE tags SET name = ?2 WHERE id = ?1", params![id, name])?;
    Ok(())
}

pub fn set_tag_hotkey(conn: &Connection, id: i64, hotkey: Option<i64>) -> Result<()> {
    if let Some(h) = hotkey {
        if !(1..=9).contains(&h) {
            return Err(GccfError::msg("hotkey must be 1..9"));
        }
        conn.execute("UPDATE tags SET hotkey = NULL WHERE hotkey = ?1", params![h])?;
    }
    conn.execute("UPDATE tags SET hotkey = ?2 WHERE id = ?1", params![id, hotkey])?;
    Ok(())
}

pub fn sample_tag_ids(conn: &Connection, sample_id: i64) -> rusqlite::Result<Vec<i64>> {
    let mut stmt = conn.prepare_cached("SELECT tag_id FROM sample_tags WHERE sample_id = ?1 ORDER BY tag_id")?;
    let rows = stmt.query_map(params![sample_id], |r| r.get(0))?;
    rows.collect()
}

pub fn sample_tag_names(conn: &Connection, sample_id: i64) -> Result<Vec<String>> {
    let mut stmt = conn.prepare_cached(
        "SELECT t.name FROM sample_tags st JOIN tags t ON t.id = st.tag_id WHERE st.sample_id = ?1 ORDER BY t.name",
    )?;
    let rows = stmt.query_map(params![sample_id], |r| r.get(0))?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn set_sample_tag(conn: &Connection, sample_id: i64, tag_id: i64, on: bool, source: &str) -> Result<()> {
    if on {
        conn.execute(
            "INSERT OR REPLACE INTO sample_tags(sample_id, tag_id, source) VALUES(?1, ?2, ?3)",
            params![sample_id, tag_id, source],
        )?;
    } else {
        conn.execute(
            "DELETE FROM sample_tags WHERE sample_id = ?1 AND tag_id = ?2",
            params![sample_id, tag_id],
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Classifier runs & predictions
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn insert_run(
    conn: &Connection,
    session_id: Option<i64>,
    backend: &str,
    model_id: &str,
    command: Option<&str>,
    workers: i64,
    batch_size: i64,
    total: i64,
    model_row_id: Option<i64>,
    mode: &str,
) -> Result<i64> {
    conn.execute(
        "INSERT INTO classifier_runs(session_id, backend, model_id, command, workers, batch_size, status, total, started_at, model_row_id, mode)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, 'running', ?7, ?8, ?9, ?10)",
        params![session_id, backend, model_id, command, workers, batch_size, total, now_iso(), model_row_id, mode],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn update_run_progress(conn: &Connection, run_id: i64, completed: i64, failed: i64) -> Result<()> {
    conn.execute(
        "UPDATE classifier_runs SET completed = ?2, failed = ?3 WHERE id = ?1",
        params![run_id, completed, failed],
    )?;
    Ok(())
}

pub fn finish_run(conn: &Connection, run_id: i64, status: &str, error: Option<&str>) -> Result<()> {
    conn.execute(
        "UPDATE classifier_runs SET status = ?2, error = ?3, finished_at = ?4 WHERE id = ?1",
        params![run_id, status, error, now_iso()],
    )?;
    Ok(())
}

pub struct NewPrediction {
    pub sample_id: i64,
    pub verdict: String,
    pub score: f64,
    pub tags: Vec<String>,
    pub latency_ms: Option<i64>,
}

pub fn insert_predictions(conn: &mut Connection, run_id: i64, preds: &[NewPrediction]) -> Result<()> {
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare_cached(
            "INSERT INTO predictions(sample_id, run_id, verdict, score, tags_json, latency_ms, created_at)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;
        let now = now_iso();
        for p in preds {
            stmt.execute(params![
                p.sample_id,
                run_id,
                p.verdict,
                p.score,
                serde_json::to_string(&p.tags)?,
                p.latency_ms,
                now
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}

pub fn latest_prediction(conn: &Connection, sample_id: i64) -> rusqlite::Result<Option<PredictionView>> {
    let mut stmt = conn.prepare_cached(
        "SELECT p.run_id, r.model_id, p.verdict, p.score, p.tags_json, p.created_at
         FROM predictions p JOIN classifier_runs r ON r.id = p.run_id
         WHERE p.sample_id = ?1 ORDER BY p.id DESC LIMIT 1",
    )?;
    stmt.query_row(params![sample_id], |r| {
        let tags_json: String = r.get(4)?;
        Ok(PredictionView {
            run_id: r.get(0)?,
            model_id: r.get(1)?,
            verdict: r.get(2)?,
            score: r.get(3)?,
            tags: serde_json::from_str(&tags_json).unwrap_or_default(),
            created_at: r.get(5)?,
        })
    })
    .optional()
}

pub fn list_runs(conn: &Connection) -> Result<Vec<ClassifierRunInfo>> {
    let mut stmt = conn.prepare(
        "SELECT r.id, r.session_id, s.name, r.backend, r.model_id, r.command, r.workers, r.batch_size, r.status, r.total, r.completed, r.failed,
                r.started_at, r.finished_at, r.error, r.mode, r.model_row_id,
                (SELECT COUNT(*) FROM eval_feedback e WHERE e.run_id = r.id),
                (SELECT COUNT(*) FROM eval_feedback e WHERE e.run_id = r.id AND e.predicted = e.human)
         FROM classifier_runs r LEFT JOIN sessions s ON s.id = r.session_id ORDER BY r.id DESC LIMIT 200",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok(ClassifierRunInfo {
            id: r.get(0)?,
            mode: r.get::<_, Option<String>>(15)?.unwrap_or_else(|| "score".into()),
            model_row_id: r.get(16)?,
            evaluated: r.get(17)?,
            eval_agree: r.get(18)?,
            session_id: r.get(1)?,
            session_name: r.get(2)?,
            backend: r.get(3)?,
            model_id: r.get(4)?,
            command: r.get(5)?,
            workers: r.get(6)?,
            batch_size: r.get(7)?,
            status: r.get(8)?,
            total: r.get(9)?,
            completed: r.get(10)?,
            failed: r.get(11)?,
            started_at: r.get(12)?,
            finished_at: r.get(13)?,
            error: r.get(14)?,
            agreement: None,
        })
    })?;
    let mut out = rows.collect::<std::result::Result<Vec<_>, _>>()?;
    for run in &mut out {
        run.agreement = Some(run_agreement(conn, run.id)?);
    }
    Ok(out)
}

pub fn run_agreement(conn: &Connection, run_id: i64) -> Result<AgreementStats> {
    let mut stmt = conn.prepare_cached(
        "SELECT p.verdict, s.status FROM predictions p JOIN samples s ON s.id = p.sample_id
         WHERE p.run_id = ?1 AND s.status IN ('good','bad') AND s.decided_by = 'human'",
    )?;
    let rows = stmt.query_map(params![run_id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
    let mut a = AgreementStats::default();
    for row in rows {
        let (pred, truth) = row?;
        a.compared += 1;
        match (pred.as_str(), truth.as_str()) {
            ("good", "good") => {
                a.tp += 1;
                a.agree += 1;
            }
            ("bad", "bad") => {
                a.tn += 1;
                a.agree += 1;
            }
            ("good", "bad") => a.fp += 1,
            ("bad", "good") => a.fn_ += 1,
            _ => {}
        }
    }
    Ok(a)
}

/// Samples of a session that a classifier run should score.
pub fn samples_for_classification(conn: &Connection, session_id: Option<i64>, only_pending: bool) -> Result<Vec<(i64, String, String, String)>> {
    let mut sql = String::from(
        "SELECT s.id, s.text, f.rel_path, f.format FROM samples s JOIN files f ON f.id = s.file_id WHERE 1=1",
    );
    let mut args: Vec<rusqlite::types::Value> = Vec::new();
    if let Some(sid) = session_id {
        sql.push_str(" AND s.session_id = ?");
        args.push(sid.into());
    }
    if only_pending {
        sql.push_str(" AND s.status = 'pending'");
    }
    sql.push_str(" ORDER BY s.session_id, s.ordinal");
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(args.iter()), |r| {
        Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

/// Apply a run's predictions as classifier-sourced decisions on samples that
/// are still pending (never overrides a human decision). Returns the count.
pub fn apply_run_predictions(conn: &mut Connection, run_id: i64, min_score: f64, apply_tags: bool) -> Result<i64> {
    let preds: Vec<(i64, String, f64, String)> = {
        let mut stmt = conn.prepare(
            "SELECT p.sample_id, p.verdict, p.score, p.tags_json FROM predictions p JOIN samples s ON s.id = p.sample_id
             WHERE p.run_id = ?1 AND s.status = 'pending' AND p.score >= ?2 AND p.verdict IN ('good','bad')",
        )?;
        let rows = stmt.query_map(params![run_id, min_score], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };
    let model_id: String = conn.query_row(
        "SELECT model_id FROM classifier_runs WHERE id = ?1",
        params![run_id],
        |r| r.get(0),
    )?;
    let mut n = 0;
    let tx = conn.transaction()?;
    for (sample_id, verdict, score, tags_json) in preds {
        if apply_tags {
            let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
            let ids = get_or_create_tag_ids(&tx, &tags)?;
            for tid in ids {
                set_sample_tag(&tx, sample_id, tid, true, "classifier")?;
            }
        }
        decide(&tx, sample_id, &verdict, "classifier", Some(&model_id), Some(score), None)?;
        n += 1;
    }
    tx.commit()?;
    Ok(n)
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

pub fn get_setting(conn: &Connection, key: &str) -> Result<Option<String>> {
    Ok(conn
        .query_row("SELECT value FROM settings WHERE key = ?1", params![key], |r| r.get(0))
        .optional()?)
}

pub fn set_setting(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO settings(key, value) VALUES(?1, ?2)",
        params![key, value],
    )?;
    Ok(())
}

pub fn all_settings(conn: &Connection) -> Result<HashMap<String, String>> {
    let mut stmt = conn.prepare("SELECT key, value FROM settings")?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
    Ok(rows.collect::<std::result::Result<HashMap<_, _>, _>>()?)
}

// ---------------------------------------------------------------------------
// Export helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ExportSampleRow {
    pub sample_id: i64,
    pub session_id: i64,
    pub root_path: String,
    pub rel_path: String,
    pub format: String,
    pub kind: String,
    pub compression: String,
    pub locator_kind: String,
    pub locator: serde_json::Value,
    pub content_hash: String,
    pub status: String,
    pub decided_at: Option<String>,
    pub decided_by: Option<String>,
    pub tags: Vec<String>,
    pub text: String,
    pub text_bytes: i64,
    pub truncated: bool,
    /// Most recent evaluation feedback on this sample, if a model was
    /// evaluated against it (what it predicted vs. what the human said).
    pub eval: Option<EvalFeedbackView>,
}

/// Stream samples (with their tags) through a callback, in ordinal order.
pub fn for_each_sample_export<F>(
    conn: &Connection,
    session_ids: &[i64],
    statuses: &[&str],
    mut f: F,
) -> Result<i64>
where
    F: FnMut(ExportSampleRow) -> Result<()>,
{
    let status_ph = statuses.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let mut sql = format!(
        "SELECT s.id, s.session_id, r.path, fi.rel_path, fi.format, fi.kind, fi.compression, s.locator_kind, s.locator_json,
                s.content_hash, s.status, s.decided_at, s.decided_by, s.text, s.text_bytes, s.truncated
         FROM samples s JOIN files fi ON fi.id = s.file_id JOIN roots r ON r.id = fi.root_id
         WHERE s.status IN ({status_ph})"
    );
    let mut args: Vec<rusqlite::types::Value> = statuses.iter().map(|s| (*s).to_string().into()).collect();
    if !session_ids.is_empty() {
        let ph = session_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        sql.push_str(&format!(" AND s.session_id IN ({ph})"));
        for id in session_ids {
            args.push((*id).into());
        }
    }
    sql.push_str(" ORDER BY s.session_id, s.ordinal");
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query(params_from_iter(args.iter()))?;
    let mut n = 0;
    while let Some(r) = rows.next()? {
        let id: i64 = r.get(0)?;
        let locator_json: String = r.get(8)?;
        let row = ExportSampleRow {
            sample_id: id,
            session_id: r.get(1)?,
            root_path: r.get(2)?,
            rel_path: r.get(3)?,
            format: r.get(4)?,
            kind: r.get(5)?,
            compression: r.get(6)?,
            locator_kind: r.get(7)?,
            locator: serde_json::from_str(&locator_json).unwrap_or(serde_json::Value::Null),
            content_hash: r.get(9)?,
            status: r.get(10)?,
            decided_at: r.get(11)?,
            decided_by: r.get(12)?,
            tags: sample_tag_names(conn, id)?,
            text: r.get(13)?,
            text_bytes: r.get(14)?,
            truncated: r.get::<_, i64>(15)? != 0,
            eval: latest_eval_feedback(conn, id)?,
        };
        f(row)?;
        n += 1;
    }
    Ok(n)
}

// ---------------------------------------------------------------------------
// Models (trained classifiers)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: i64,
    pub name: String,
    /// "builtin" (served in-process) | "external" (served by `command`)
    pub kind: String,
    pub path: String,
    pub command: Option<String>,
    pub parent_id: Option<i64>,
    pub parent_name: Option<String>,
    pub created_at: String,
    pub train_rows: i64,
    pub metrics: serde_json::Value,
    pub tag_names: Vec<String>,
    pub notes: Option<String>,
    pub runs: i64,
    /// Latest evaluation run of this model, if any.
    pub latest_eval: Option<EvalBrief>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalBrief {
    pub run_id: i64,
    pub evaluated: i64,
    pub agree: i64,
    pub pending: i64,
}

pub struct NewModel<'a> {
    pub name: &'a str,
    pub kind: &'a str,
    pub path: &'a str,
    pub command: Option<&'a str>,
    pub parent_id: Option<i64>,
    pub train_rows: i64,
    pub metrics: &'a serde_json::Value,
    pub tag_names: &'a [String],
    pub notes: Option<&'a str>,
}

pub fn insert_model(conn: &Connection, m: &NewModel<'_>) -> Result<i64> {
    conn.execute(
        "INSERT INTO models(name, kind, path, command, parent_id, created_at, train_rows, metrics_json, tag_names_json, notes)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            m.name,
            m.kind,
            m.path,
            m.command,
            m.parent_id,
            now_iso(),
            m.train_rows,
            serde_json::to_string(m.metrics)?,
            serde_json::to_string(m.tag_names)?,
            m.notes
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

fn model_from(conn: &Connection, r: &Row<'_>) -> rusqlite::Result<ModelInfo> {
    let id: i64 = r.get(0)?;
    let metrics: Option<String> = r.get(8)?;
    let tags: Option<String> = r.get(9)?;
    let latest_eval = conn
        .query_row(
            "SELECT r.id,
                    (SELECT COUNT(*) FROM eval_feedback e WHERE e.run_id = r.id AND e.human != 'skip'),
                    (SELECT COUNT(*) FROM eval_feedback e WHERE e.run_id = r.id AND e.predicted = e.human),
                    (SELECT COUNT(*) FROM predictions p JOIN samples s ON s.id = p.sample_id WHERE p.run_id = r.id AND s.status = 'pending')
             FROM classifier_runs r WHERE r.model_row_id = ?1 AND r.mode = 'eval' ORDER BY r.id DESC LIMIT 1",
            params![id],
            |r| {
                Ok(EvalBrief {
                    run_id: r.get(0)?,
                    evaluated: r.get(1)?,
                    agree: r.get(2)?,
                    pending: r.get(3)?,
                })
            },
        )
        .optional()?;
    Ok(ModelInfo {
        id,
        name: r.get(1)?,
        kind: r.get(2)?,
        path: r.get(3)?,
        command: r.get(4)?,
        parent_id: r.get(5)?,
        parent_name: r.get(6)?,
        created_at: r.get(7)?,
        metrics: metrics.and_then(|m| serde_json::from_str(&m).ok()).unwrap_or(serde_json::Value::Null),
        tag_names: tags.and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default(),
        train_rows: r.get(10)?,
        notes: r.get(11)?,
        runs: r.get(12)?,
        latest_eval,
    })
}

const MODEL_SELECT: &str = "SELECT m.id, m.name, m.kind, m.path, m.command, m.parent_id, p.name, m.created_at, m.metrics_json, m.tag_names_json, m.train_rows, m.notes,
        (SELECT COUNT(*) FROM classifier_runs r WHERE r.model_row_id = m.id)
        FROM models m LEFT JOIN models p ON p.id = m.parent_id";

pub fn list_models(conn: &Connection) -> Result<Vec<ModelInfo>> {
    let mut stmt = conn.prepare(&format!("{MODEL_SELECT} ORDER BY m.id DESC"))?;
    let rows = stmt.query_map([], |r| model_from(conn, r))?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn get_model(conn: &Connection, id: i64) -> Result<ModelInfo> {
    let mut stmt = conn.prepare(&format!("{MODEL_SELECT} WHERE m.id = ?1"))?;
    stmt.query_row(params![id], |r| model_from(conn, r))
        .optional()?
        .ok_or_else(|| GccfError::msg(format!("model {id} not found")))
}

pub fn delete_model(conn: &Connection, id: i64) -> Result<()> {
    conn.execute("DELETE FROM models WHERE id = ?1", params![id])?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Evaluation feedback
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalFeedbackView {
    pub run_id: i64,
    pub sample_id: i64,
    pub predicted: String,
    pub score: f64,
    pub predicted_tags: Vec<String>,
    pub human: String,
    pub human_tags: Vec<String>,
    pub agree: bool,
    pub blind: bool,
    pub created_at: String,
}

pub fn prediction_for_run(conn: &Connection, run_id: i64, sample_id: i64) -> rusqlite::Result<Option<PredictionView>> {
    let mut stmt = conn.prepare_cached(
        "SELECT p.run_id, r.model_id, p.verdict, p.score, p.tags_json, p.created_at
         FROM predictions p JOIN classifier_runs r ON r.id = p.run_id
         WHERE p.run_id = ?1 AND p.sample_id = ?2 ORDER BY p.id DESC LIMIT 1",
    )?;
    stmt.query_row(params![run_id, sample_id], |r| {
        let tags_json: String = r.get(4)?;
        Ok(PredictionView {
            run_id: r.get(0)?,
            model_id: r.get(1)?,
            verdict: r.get(2)?,
            score: r.get(3)?,
            tags: serde_json::from_str(&tags_json).unwrap_or_default(),
            created_at: r.get(5)?,
        })
    })
    .optional()
}

/// Next sample to evaluate for a run: still pending, has a prediction from
/// this run, ordered by ordinal. The returned view carries *this run's*
/// prediction rather than the latest one.
pub fn next_eval_sample(conn: &Connection, run_id: i64, after_ordinal: Option<i64>) -> Result<Option<SampleView>> {
    let base = format!(
        "{SAMPLE_SELECT} WHERE s.status = 'pending' AND EXISTS (SELECT 1 FROM predictions p WHERE p.run_id = ?1 AND p.sample_id = s.id)"
    );
    let mut found: Option<SampleView> = None;
    if let Some(after) = after_ordinal {
        let mut stmt = conn.prepare_cached(&format!("{base} AND s.ordinal > ?2 ORDER BY s.session_id, s.ordinal LIMIT 1"))?;
        found = stmt.query_row(params![run_id, after], |r| sample_view_from(conn, r)).optional()?;
    }
    if found.is_none() {
        let mut stmt = conn.prepare_cached(&format!("{base} ORDER BY s.session_id, s.ordinal LIMIT 1"))?;
        found = stmt.query_row(params![run_id], |r| sample_view_from(conn, r)).optional()?;
    }
    if let Some(v) = found.as_mut() {
        v.prediction = prediction_for_run(conn, run_id, v.id)?;
    }
    Ok(found)
}

/// Record what the human decided about a sample that a run had predicted.
/// Returns `None` when the run never scored this sample.
pub fn insert_eval_feedback(conn: &Connection, run_id: i64, sample_id: i64, human: &str, blind: bool, view_ms: Option<i64>) -> Result<Option<EvalFeedbackView>> {
    let Some(pred) = prediction_for_run(conn, run_id, sample_id)? else {
        return Ok(None);
    };
    let human_tags = sample_tag_names(conn, sample_id)?;
    let now = now_iso();
    conn.execute(
        "INSERT INTO eval_feedback(run_id, sample_id, predicted, score, predicted_tags_json, human, human_tags_json, blind, view_ms, created_at)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
         ON CONFLICT(run_id, sample_id) DO UPDATE SET human = excluded.human, human_tags_json = excluded.human_tags_json,
           blind = excluded.blind, view_ms = excluded.view_ms, created_at = excluded.created_at",
        params![
            run_id,
            sample_id,
            pred.verdict,
            pred.score,
            serde_json::to_string(&pred.tags)?,
            human,
            serde_json::to_string(&human_tags)?,
            blind as i64,
            view_ms,
            now
        ],
    )?;
    Ok(Some(EvalFeedbackView {
        run_id,
        sample_id,
        agree: pred.verdict == human,
        predicted: pred.verdict,
        score: pred.score,
        predicted_tags: pred.tags,
        human: human.to_string(),
        human_tags,
        blind,
        created_at: now,
    }))
}

pub fn latest_eval_feedback(conn: &Connection, sample_id: i64) -> rusqlite::Result<Option<EvalFeedbackView>> {
    let mut stmt = conn.prepare_cached(
        "SELECT run_id, sample_id, predicted, score, predicted_tags_json, human, human_tags_json, blind, created_at
         FROM eval_feedback WHERE sample_id = ?1 ORDER BY id DESC LIMIT 1",
    )?;
    stmt.query_row(params![sample_id], |r| {
        let pt: String = r.get(4)?;
        let ht: String = r.get(6)?;
        let predicted: String = r.get(2)?;
        let human: String = r.get(5)?;
        Ok(EvalFeedbackView {
            run_id: r.get(0)?,
            sample_id: r.get(1)?,
            agree: predicted == human,
            predicted,
            score: r.get(3)?,
            predicted_tags: serde_json::from_str(&pt).unwrap_or_default(),
            human,
            human_tags: serde_json::from_str(&ht).unwrap_or_default(),
            blind: r.get::<_, i64>(7)? != 0,
            created_at: r.get(8)?,
        })
    })
    .optional()
}

/// sample_id → true when some evaluation of it disagreed with the human.
/// Used to up-weight hard cases when finetuning.
pub fn eval_disagreement_samples(conn: &Connection) -> Result<HashMap<i64, bool>> {
    let mut stmt = conn.prepare(
        "SELECT sample_id, MIN(CASE WHEN predicted = human THEN 1 ELSE 0 END) FROM eval_feedback WHERE human != 'skip' GROUP BY sample_id",
    )?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)? == 0)))?;
    Ok(rows.collect::<std::result::Result<HashMap<_, _>, _>>()?)
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CalibrationBin {
    pub lo: f64,
    pub hi: f64,
    pub n: i64,
    pub agree: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TagAgreement {
    pub name: String,
    pub predicted: i64,
    pub human: i64,
    pub both: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Disagreement {
    pub sample_id: i64,
    pub session_id: i64,
    pub rel_path: String,
    pub predicted: String,
    pub score: f64,
    pub human: String,
    pub preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvalReport {
    pub run_id: i64,
    pub model_row_id: Option<i64>,
    pub model_name: String,
    pub scored: i64,
    pub remaining: i64,
    pub evaluated: i64,
    pub agree: i64,
    pub blind: i64,
    pub tp: i64,
    pub fp: i64,
    pub tn: i64,
    pub fn_: i64,
    pub accuracy: Option<f64>,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
    pub calibration: Vec<CalibrationBin>,
    pub tags: Vec<TagAgreement>,
    pub disagreements: Vec<Disagreement>,
    pub gate_min_accuracy: f64,
    pub gate_min_evaluated: i64,
    pub capable: bool,
}

pub fn eval_report(conn: &Connection, run_id: i64, gate_min_accuracy: f64, gate_min_evaluated: i64) -> Result<EvalReport> {
    let (model_row_id, model_name, scored): (Option<i64>, String, i64) = conn.query_row(
        "SELECT model_row_id, model_id, (SELECT COUNT(*) FROM predictions p WHERE p.run_id = r.id) FROM classifier_runs r WHERE r.id = ?1",
        params![run_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?;
    let remaining: i64 = conn.query_row(
        "SELECT COUNT(*) FROM predictions p JOIN samples s ON s.id = p.sample_id WHERE p.run_id = ?1 AND s.status = 'pending'",
        params![run_id],
        |r| r.get(0),
    )?;
    let mut rep = EvalReport {
        run_id,
        model_row_id,
        model_name,
        scored,
        remaining,
        gate_min_accuracy,
        gate_min_evaluated,
        ..Default::default()
    };
    let mut bins: Vec<CalibrationBin> = (0..5)
        .map(|i| CalibrationBin {
            lo: 0.5 + i as f64 * 0.1,
            hi: 0.6 + i as f64 * 0.1,
            ..Default::default()
        })
        .collect();
    let mut tag_stats: HashMap<String, TagAgreement> = HashMap::new();
    {
        let mut stmt = conn.prepare(
            "SELECT predicted, score, human, blind, predicted_tags_json, human_tags_json FROM eval_feedback WHERE run_id = ?1",
        )?;
        let rows = stmt.query_map(params![run_id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, f64>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, String>(5)?,
            ))
        })?;
        for row in rows {
            let (predicted, score, human, blind, pt, ht) = row?;
            if human == "skip" {
                continue; // undecided: no signal
            }
            rep.evaluated += 1;
            rep.blind += blind;
            let agree = predicted == human;
            if agree {
                rep.agree += 1;
            }
            match (predicted.as_str(), human.as_str()) {
                ("good", "good") => rep.tp += 1,
                ("good", "bad") => rep.fp += 1,
                ("bad", "bad") => rep.tn += 1,
                ("bad", "good") => rep.fn_ += 1,
                _ => {}
            }
            let bi = (((score.clamp(0.5, 0.9999) - 0.5) / 0.1) as usize).min(4);
            bins[bi].n += 1;
            if agree {
                bins[bi].agree += 1;
            }
            let ptags: Vec<String> = serde_json::from_str(&pt).unwrap_or_default();
            let htags: Vec<String> = serde_json::from_str(&ht).unwrap_or_default();
            for t in &ptags {
                let e = tag_stats.entry(t.clone()).or_insert_with(|| TagAgreement { name: t.clone(), ..Default::default() });
                e.predicted += 1;
                if htags.contains(t) {
                    e.both += 1;
                }
            }
            for t in &htags {
                let e = tag_stats.entry(t.clone()).or_insert_with(|| TagAgreement { name: t.clone(), ..Default::default() });
                e.human += 1;
            }
        }
    }
    if rep.evaluated > 0 {
        rep.accuracy = Some(rep.agree as f64 / rep.evaluated as f64);
    }
    if rep.tp + rep.fp > 0 {
        rep.precision = Some(rep.tp as f64 / (rep.tp + rep.fp) as f64);
    }
    if rep.tp + rep.fn_ > 0 {
        rep.recall = Some(rep.tp as f64 / (rep.tp + rep.fn_) as f64);
    }
    if let (Some(p), Some(r)) = (rep.precision, rep.recall) {
        if p + r > 0.0 {
            rep.f1 = Some(2.0 * p * r / (p + r));
        }
    }
    rep.calibration = bins;
    let mut tags: Vec<TagAgreement> = tag_stats.into_values().collect();
    tags.sort_by(|a, b| (b.human + b.predicted).cmp(&(a.human + a.predicted)));
    rep.tags = tags;
    rep.capable = rep.evaluated >= gate_min_evaluated && rep.accuracy.map(|a| a >= gate_min_accuracy).unwrap_or(false);

    let mut stmt = conn.prepare(
        "SELECT e.sample_id, s.session_id, f.rel_path, e.predicted, e.score, e.human, substr(s.text, 1, 200)
         FROM eval_feedback e JOIN samples s ON s.id = e.sample_id JOIN files f ON f.id = s.file_id
         WHERE e.run_id = ?1 AND e.predicted != e.human AND e.human != 'skip' ORDER BY e.score DESC LIMIT 200",
    )?;
    let rows = stmt.query_map(params![run_id], |r| {
        Ok(Disagreement {
            sample_id: r.get(0)?,
            session_id: r.get(1)?,
            rel_path: r.get(2)?,
            predicted: r.get(3)?,
            score: r.get(4)?,
            human: r.get(5)?,
            preview: r.get(6)?,
        })
    })?;
    rep.disagreements = rows.collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_and_tags() {
        let db = Db::open_memory().unwrap();
        let conn = db.lock();
        let tags = list_tags(&conn).unwrap();
        assert!(tags.iter().any(|t| t.name == "code" && t.hotkey == Some(1)));
        let id = create_tag(&conn, "  Native English!! ").unwrap();
        let again = create_tag(&conn, "native-english").unwrap();
        assert_eq!(id, again);
    }

    #[test]
    fn decide_flow() {
        let db = Db::open_memory().unwrap();
        let mut conn = db.lock();
        let root = insert_root(&conn, "C:/corpora", None).unwrap();
        let gen = begin_scan_generation(&conn, root).unwrap();
        upsert_files(
            &mut conn,
            root,
            gen,
            &[ScannedFile {
                rel_path: "a.jsonl".into(),
                size_bytes: 10,
                mtime: 0,
                ext: "jsonl".into(),
                format: Format::Jsonl,
                kind: Kind::Rows,
                compression: Compression::None,
            }],
        )
        .unwrap();
        finish_scan(&conn, root, gen).unwrap();
        let file_id = eligible_file_ids(&conn, root, &[], true).unwrap()[0];
        let sid = insert_session(
            &conn,
            &NewSession {
                root_id: root,
                name: "s1",
                files_requested: 1,
                samples_per_file: 1,
                lines_per_sample: 1,
                seed: 1,
                format_filter: &[],
                exclude_reviewed_files: true,
            },
        )
        .unwrap();
        insert_samples(
            &mut conn,
            sid,
            0,
            &[NewSample {
                file_id,
                locator_kind: "bytes",
                locator_json: "{}",
                text: "hello",
                truncated: false,
                content_hash: "abc",
            }],
        )
        .unwrap();
        finish_session(&conn, sid, 1, "ready", None).unwrap();
        let s = next_sample(&conn, sid, &["pending"], None).unwrap().unwrap();
        let tag = create_tag(&conn, "english").unwrap();
        set_sample_tag(&conn, s.id, tag, true, "human").unwrap();
        decide(&conn, s.id, "good", "human", None, None, Some(1200)).unwrap();
        assert!(next_sample(&conn, sid, &["pending"], None).unwrap().is_none());
        let counts = session_counts(&conn, sid).unwrap();
        assert_eq!(counts.good, 1);
        // file now reviewed => excluded from next session
        assert!(eligible_file_ids(&conn, root, &[], true).unwrap().is_empty());
        assert_eq!(eligible_file_ids(&conn, root, &[], false).unwrap().len(), 1);
    }
}
