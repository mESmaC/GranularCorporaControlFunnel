//! Sample extraction. The guiding rule: cost must scale with the number of
//! samples requested, never with the size of the file, except where the
//! container makes that impossible (gzip / zstd streams, which are read once
//! with reservoir sampling and can be byte-capped).

use crate::error::{GccfError, Result};
use crate::formats::{sniff_is_text, sniff_json_is_ndjson, Compression, Format, Kind};
use crate::rng::Rng;
use memchr::memchr;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Files up to this size are read entirely and windowed exactly; larger files
/// use random byte offsets (O(samples) I/O).
pub const SMALL_FILE_LIMIT: u64 = 8 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct SampleParams {
    pub samples_per_file: usize,
    pub lines_per_sample: usize,
    pub seed: u64,
    /// Maximum bytes of text kept per sample (rest is dropped, sample flagged truncated).
    pub max_bytes_per_sample: usize,
    /// For compressed streams: stop after this many decompressed bytes (0 = unlimited).
    pub max_stream_bytes: u64,
}

impl Default for SampleParams {
    fn default() -> Self {
        SampleParams {
            samples_per_file: 50,
            lines_per_sample: 50,
            seed: 42,
            max_bytes_per_sample: 256 * 1024,
            max_stream_bytes: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Locator {
    /// A single record in a parquet file, by absolute row index.
    Row { index: i64 },
    /// A line-aligned byte range `[byte_start, byte_end)` in an uncompressed file.
    Bytes {
        byte_start: u64,
        byte_end: u64,
        n_lines: usize,
    },
    /// A window of lines inside a compressed stream (0-based line index of the
    /// first line after decompression).
    StreamLines { line_start: i64, n_lines: usize },
}

impl Locator {
    pub fn kind(&self) -> &'static str {
        match self {
            Locator::Row { .. } => "row",
            Locator::Bytes { .. } => "bytes",
            Locator::StreamLines { .. } => "stream_lines",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExtractedSample {
    pub locator: Locator,
    pub text: String,
    pub truncated: bool,
}

#[derive(Debug, Default)]
pub struct SampleOutput {
    pub samples: Vec<ExtractedSample>,
    pub row_count: Option<i64>,
    /// Effective format / kind after sniffing (may differ from catalogue).
    pub effective_format: Option<Format>,
    pub effective_kind: Option<Kind>,
    pub note: Option<String>,
}

pub fn content_hash(text: &str) -> String {
    blake3::hash(text.as_bytes()).to_hex().to_string()
}

fn truncate_utf8(bytes: &[u8], cap: usize) -> (String, bool) {
    if bytes.len() <= cap {
        return (String::from_utf8_lossy(bytes).into_owned(), false);
    }
    let s = String::from_utf8_lossy(&bytes[..cap]);
    // from_utf8_lossy on a cut boundary yields a replacement char at the end; trim it.
    let mut s = s.into_owned();
    while s.ends_with('\u{FFFD}') {
        s.pop();
    }
    (s, true)
}

/// Entry point: sample one file according to its catalogued classification.
pub fn sample_file(
    path: &Path,
    format: Format,
    kind: Kind,
    compression: Compression,
    params: &SampleParams,
) -> Result<SampleOutput> {
    if kind == Kind::Unsupported || params.samples_per_file == 0 {
        return Ok(SampleOutput::default());
    }
    let mut rng = Rng::derive(params.seed, &path.to_string_lossy());
    let k = params.samples_per_file;

    match compression {
        Compression::Unsupported => Ok(SampleOutput {
            note: Some("unsupported compression".into()),
            ..Default::default()
        }),
        Compression::None => {
            if format == Format::Parquet {
                return sample_parquet(path, k, &mut rng, params.max_bytes_per_sample);
            }
            let (eff_format, eff_kind) = resniff(path, format, kind)?;
            if eff_kind == Kind::Unsupported {
                return Ok(SampleOutput {
                    effective_format: Some(eff_format),
                    effective_kind: Some(eff_kind),
                    note: Some("binary content".into()),
                    ..Default::default()
                });
            }
            let window = if eff_kind == Kind::Rows { 1 } else { params.lines_per_sample.max(1) };
            let has_header = matches!(eff_format, Format::Csv | Format::Tsv);
            let mut out = sample_uncompressed(path, window, k, &mut rng, params.max_bytes_per_sample, has_header, SMALL_FILE_LIMIT)?;
            out.effective_format = Some(eff_format);
            out.effective_kind = Some(eff_kind);
            Ok(out)
        }
        Compression::Gzip | Compression::Zstd => {
            let window = if kind == Kind::Rows { 1 } else { params.lines_per_sample.max(1) };
            let has_header = matches!(format, Format::Csv | Format::Tsv);
            let file = File::open(path)?;
            let reader: Box<dyn Read> = match compression {
                Compression::Gzip => Box::new(flate2::read::MultiGzDecoder::new(BufReader::with_capacity(1 << 16, file))),
                _ => Box::new(zstd::stream::read::Decoder::new(file).map_err(GccfError::Io)?),
            };
            let mut reader = BufReader::with_capacity(1 << 16, reader);
            let mut out = sample_stream(&mut reader, window, k, &mut rng, params.max_bytes_per_sample, has_header, params.max_stream_bytes)?;
            out.effective_format = Some(format);
            out.effective_kind = Some(kind);
            Ok(out)
        }
    }
}

/// Second look at ambiguous formats now that we are willing to open the file.
fn resniff(path: &Path, format: Format, kind: Kind) -> Result<(Format, Kind)> {
    match format {
        Format::Json => {
            let mut f = File::open(path)?;
            let mut head = vec![0u8; 4096];
            let n = f.read(&mut head)?;
            head.truncate(n);
            if head.contains(&0u8) {
                return Ok((Format::Binary, Kind::Unsupported));
            }
            if sniff_json_is_ndjson(&head) {
                Ok((Format::Jsonl, Kind::Rows))
            } else {
                Ok((Format::Json, Kind::Lines))
            }
        }
        Format::Unknown => match sniff_is_text(path) {
            Some(true) => Ok((Format::Text, Kind::Lines)),
            _ => Ok((Format::Binary, Kind::Unsupported)),
        },
        _ => Ok((format, kind)),
    }
}

// ---------------------------------------------------------------------------
// Line reading helpers (memory-bounded)
// ---------------------------------------------------------------------------

struct LineRead {
    /// Bytes consumed from the reader, including the newline if one was found.
    consumed: usize,
    found_newline: bool,
}

fn append_capped(out: &mut Vec<u8>, src: &[u8], cap: usize, truncated: &mut bool) {
    let room = cap.saturating_sub(out.len());
    if src.len() <= room {
        out.extend_from_slice(src);
    } else {
        out.extend_from_slice(&src[..room]);
        *truncated = true;
    }
}

/// Read one line (without its newline) appending at most `cap` bytes total
/// into `out`. Always consumes the whole line from the reader.
fn read_line_capped<R: BufRead>(r: &mut R, cap: usize, out: &mut Vec<u8>, truncated: &mut bool) -> std::io::Result<LineRead> {
    let mut consumed = 0usize;
    loop {
        let buf = r.fill_buf()?;
        if buf.is_empty() {
            return Ok(LineRead {
                consumed,
                found_newline: false,
            });
        }
        match memchr(b'\n', buf) {
            Some(i) => {
                append_capped(out, &buf[..i], cap, truncated);
                consumed += i + 1;
                r.consume(i + 1);
                return Ok(LineRead {
                    consumed,
                    found_newline: true,
                });
            }
            None => {
                let n = buf.len();
                append_capped(out, buf, cap, truncated);
                consumed += n;
                r.consume(n);
            }
        }
    }
}

/// Discard bytes up to and including the next newline. Returns bytes consumed
/// and whether a newline was found (false => EOF).
fn skip_to_newline<R: BufRead>(r: &mut R) -> std::io::Result<(usize, bool)> {
    let mut consumed = 0usize;
    loop {
        let buf = r.fill_buf()?;
        if buf.is_empty() {
            return Ok((consumed, false));
        }
        match memchr(b'\n', buf) {
            Some(i) => {
                consumed += i + 1;
                r.consume(i + 1);
                return Ok((consumed, true));
            }
            None => {
                let n = buf.len();
                consumed += n;
                r.consume(n);
            }
        }
    }
}

struct Window {
    text: Vec<u8>,
    n_lines: usize,
    consumed: usize,
    truncated: bool,
}

/// Read up to `n_lines` lines starting at the reader's current position.
fn read_window<R: BufRead>(r: &mut R, n_lines: usize, cap: usize) -> std::io::Result<Window> {
    let mut text = Vec::new();
    let mut truncated = false;
    let mut consumed = 0usize;
    let mut lines = 0usize;
    for _ in 0..n_lines {
        let before = text.len();
        let lr = read_line_capped(r, cap, &mut text, &mut truncated)?;
        if lr.consumed == 0 {
            break; // EOF
        }
        consumed += lr.consumed;
        lines += 1;
        if text.len() > before && text.ends_with(b"\r") {
            text.pop();
        }
        if !lr.found_newline {
            break;
        }
        if lines < n_lines {
            if text.len() < cap {
                text.push(b'\n');
            } else {
                truncated = true;
            }
        }
    }
    // Drop a trailing separator we may have added before hitting EOF.
    while text.ends_with(b"\n") {
        text.pop();
    }
    Ok(Window {
        text,
        n_lines: lines,
        consumed,
        truncated,
    })
}

fn read_header(path: &Path, cap: usize) -> Result<(Vec<u8>, u64)> {
    let mut r = BufReader::with_capacity(1 << 16, File::open(path)?);
    let mut text = Vec::new();
    let mut truncated = false;
    let lr = read_line_capped(&mut r, cap, &mut text, &mut truncated)?;
    if text.ends_with(b"\r") {
        text.pop();
    }
    Ok((text, lr.consumed as u64))
}

// ---------------------------------------------------------------------------
// Uncompressed files
// ---------------------------------------------------------------------------

pub fn sample_uncompressed(
    path: &Path,
    window_lines: usize,
    k: usize,
    rng: &mut Rng,
    cap: usize,
    has_header: bool,
    small_limit: u64,
) -> Result<SampleOutput> {
    let size = std::fs::metadata(path)?.len();
    if size == 0 {
        return Ok(SampleOutput::default());
    }
    let (header, header_len) = if has_header { read_header(path, cap)? } else { (Vec::new(), 0) };

    let mut samples = if size <= small_limit {
        sample_small_exact(path, window_lines, k, rng, cap, header_len)?
    } else {
        sample_large_offsets(path, size, window_lines, k, rng, cap, header_len)?
    };

    if has_header && !header.is_empty() {
        for s in &mut samples {
            let mut joined = header.clone();
            joined.push(b'\n');
            joined.extend_from_slice(s.text.as_bytes());
            let (t, tr) = truncate_utf8(&joined, cap.saturating_add(header.len() + 1));
            s.text = t;
            s.truncated |= tr;
        }
    }
    samples.sort_by_key(|s| match &s.locator {
        Locator::Bytes { byte_start, .. } => *byte_start,
        _ => 0,
    });
    Ok(SampleOutput {
        samples,
        ..Default::default()
    })
}

fn sample_small_exact(path: &Path, window_lines: usize, k: usize, rng: &mut Rng, cap: usize, skip_bytes: u64) -> Result<Vec<ExtractedSample>> {
    let data = std::fs::read(path)?;
    let start = (skip_bytes as usize).min(data.len());
    let body = &data[start..];
    // Line starts (offsets relative to body).
    let mut line_starts: Vec<usize> = vec![0];
    for nl in memchr::memchr_iter(b'\n', body) {
        if nl + 1 < body.len() {
            line_starts.push(nl + 1);
        }
    }
    if body.is_empty() {
        return Ok(Vec::new());
    }
    let n_lines = line_starts.len();
    let n_windows = (n_lines + window_lines - 1) / window_lines;
    let chosen = rng.distinct_sorted(n_windows as u64, k);
    let mut out = Vec::with_capacity(chosen.len());
    for w in chosen {
        let first = w as usize * window_lines;
        let last = ((w as usize + 1) * window_lines).min(n_lines);
        let byte_start = line_starts[first];
        let byte_end = if last < n_lines { line_starts[last] } else { body.len() };
        let mut slice = &body[byte_start..byte_end];
        // Strip trailing newline(s) / CR for the text only.
        while slice.ends_with(b"\n") || slice.ends_with(b"\r") {
            slice = &slice[..slice.len() - 1];
        }
        // Normalise CRLF inside the window.
        let cleaned: Vec<u8> = if slice.contains(&b'\r') {
            slice.iter().copied().filter(|&b| b != b'\r').collect()
        } else {
            slice.to_vec()
        };
        let (text, truncated) = truncate_utf8(&cleaned, cap);
        out.push(ExtractedSample {
            locator: Locator::Bytes {
                byte_start: start as u64 + byte_start as u64,
                byte_end: start as u64 + byte_end as u64,
                n_lines: last - first,
            },
            text,
            truncated,
        });
    }
    Ok(out)
}

fn sample_large_offsets(
    path: &Path,
    size: u64,
    window_lines: usize,
    k: usize,
    rng: &mut Rng,
    cap: usize,
    skip_bytes: u64,
) -> Result<Vec<ExtractedSample>> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(1 << 16, file);
    let mut chosen: Vec<(u64, u64)> = Vec::new(); // (start, end) ranges already taken
    let mut out = Vec::with_capacity(k);
    let max_attempts = k * 12 + 16;
    let mut attempts = 0;
    let body_start = skip_bytes.min(size);
    if body_start >= size {
        return Ok(out);
    }
    while out.len() < k && attempts < max_attempts {
        attempts += 1;
        let off = body_start + rng.below(size - body_start);
        reader.seek(SeekFrom::Start(off))?;
        let mut pos = off;
        if off > body_start {
            // We are somewhere inside a line: discard the remainder of it.
            let (consumed, found) = skip_to_newline(&mut reader)?;
            if !found {
                continue; // landed in the final partial line; retry
            }
            pos += consumed as u64;
        }
        if pos >= size {
            continue;
        }
        if chosen.iter().any(|&(s, e)| pos >= s && pos < e) {
            continue; // overlaps an existing window start
        }
        let w = read_window(&mut reader, window_lines, cap)?;
        if w.n_lines == 0 {
            continue;
        }
        let end = pos + w.consumed as u64;
        if chosen.iter().any(|&(s, e)| pos < e && end > s) {
            continue; // overlaps
        }
        chosen.push((pos, end));
        let cleaned: Vec<u8> = if w.text.contains(&b'\r') {
            w.text.iter().copied().filter(|&b| b != b'\r').collect()
        } else {
            w.text
        };
        let (text, truncated) = truncate_utf8(&cleaned, cap);
        out.push(ExtractedSample {
            locator: Locator::Bytes {
                byte_start: pos,
                byte_end: end,
                n_lines: w.n_lines,
            },
            text,
            truncated: truncated || w.truncated,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Compressed streams: reservoir sampling of windows
// ---------------------------------------------------------------------------

pub fn sample_stream<R: BufRead>(
    reader: &mut R,
    window_lines: usize,
    k: usize,
    rng: &mut Rng,
    cap: usize,
    has_header: bool,
    max_stream_bytes: u64,
) -> Result<SampleOutput> {
    struct Cand {
        line_start: i64,
        window: Window,
    }
    let mut header: Vec<u8> = Vec::new();
    if has_header {
        let mut truncated = false;
        read_line_capped(reader, cap, &mut header, &mut truncated)?;
        if header.ends_with(b"\r") {
            header.pop();
        }
    }
    let mut reservoir: Vec<Cand> = Vec::with_capacity(k);
    let mut line_idx: i64 = if has_header { 1 } else { 0 };
    let mut window_idx: u64 = 0;
    let mut bytes_seen: u64 = 0;
    let mut capped = false;
    loop {
        let w = read_window(reader, window_lines, cap)?;
        if w.n_lines == 0 {
            break;
        }
        bytes_seen += w.consumed as u64;
        let this_start = line_idx;
        line_idx += w.n_lines as i64;
        if reservoir.len() < k {
            reservoir.push(Cand { line_start: this_start, window: w });
        } else {
            let j = rng.below(window_idx + 1);
            if (j as usize) < k {
                reservoir[j as usize] = Cand { line_start: this_start, window: w };
            }
        }
        window_idx += 1;
        if max_stream_bytes > 0 && bytes_seen >= max_stream_bytes {
            capped = true;
            break;
        }
    }
    reservoir.sort_by_key(|c| c.line_start);
    let mut samples = Vec::with_capacity(reservoir.len());
    for c in reservoir {
        let mut bytes = c.window.text;
        if bytes.contains(&b'\r') {
            bytes.retain(|&b| b != b'\r');
        }
        if has_header && !header.is_empty() {
            let mut joined = header.clone();
            joined.push(b'\n');
            joined.extend_from_slice(&bytes);
            bytes = joined;
        }
        let (text, truncated) = truncate_utf8(&bytes, cap + header.len() + 1);
        samples.push(ExtractedSample {
            locator: Locator::StreamLines {
                line_start: c.line_start,
                n_lines: c.window.n_lines,
            },
            text,
            truncated: truncated || c.window.truncated,
        });
    }
    Ok(SampleOutput {
        samples,
        row_count: if capped { None } else { Some(line_idx) },
        note: if capped {
            Some(format!("stream capped after {bytes_seen} decompressed bytes"))
        } else {
            None
        },
        ..Default::default()
    })
}

// ---------------------------------------------------------------------------
// Parquet
// ---------------------------------------------------------------------------

pub fn sample_parquet(path: &Path, k: usize, rng: &mut Rng, cap: usize) -> Result<SampleOutput> {
    use arrow_schema::DataType;
    use parquet::arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection, RowSelector};
    use parquet::arrow::ProjectionMask;

    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let total = builder.metadata().file_metadata().num_rows();
    if total <= 0 {
        return Ok(SampleOutput {
            row_count: Some(0),
            ..Default::default()
        });
    }

    // Skip binary columns (images, embeddings as bytes, ...) — they are not
    // reviewable as text and can be enormous.
    let schema = builder.schema().clone();
    let keep: Vec<usize> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| {
            !matches!(
                f.data_type(),
                DataType::Binary | DataType::LargeBinary | DataType::FixedSizeBinary(_) | DataType::BinaryView
            )
        })
        .map(|(i, _)| i)
        .collect();
    let dropped = schema.fields().len() - keep.len();
    let mask = ProjectionMask::roots(builder.parquet_schema(), keep);

    let indices = rng.distinct_sorted(total as u64, k);
    let mut selectors = Vec::with_capacity(indices.len() * 2 + 1);
    let mut cursor: u64 = 0;
    for &i in &indices {
        if i > cursor {
            selectors.push(RowSelector::skip((i - cursor) as usize));
        }
        selectors.push(RowSelector::select(1));
        cursor = i + 1;
    }
    if cursor < total as u64 {
        selectors.push(RowSelector::skip((total as u64 - cursor) as usize));
    }
    let selection = RowSelection::from(selectors);

    let reader = builder
        .with_projection(mask)
        .with_row_selection(selection)
        .with_batch_size(indices.len().max(1))
        .build()?;

    let mut lines: Vec<Vec<u8>> = Vec::with_capacity(indices.len());
    for batch in reader {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut writer = arrow_json::LineDelimitedWriter::new(&mut buf);
            writer.write(&batch)?;
            writer.finish()?;
        }
        for line in buf.split(|&b| b == b'\n') {
            if !line.is_empty() {
                lines.push(line.to_vec());
            }
        }
    }

    let mut samples = Vec::with_capacity(lines.len());
    for (i, line) in lines.into_iter().enumerate() {
        let index = indices.get(i).copied().unwrap_or(0) as i64;
        let (text, truncated) = truncate_utf8(&line, cap);
        samples.push(ExtractedSample {
            locator: Locator::Row { index },
            text,
            truncated,
        });
    }
    Ok(SampleOutput {
        samples,
        row_count: Some(total),
        effective_format: Some(Format::Parquet),
        effective_kind: Some(Kind::Rows),
        note: if dropped > 0 {
            Some(format!("{dropped} binary column(s) omitted"))
        } else {
            None
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    fn p(d: &tempfile::TempDir, name: &str) -> std::path::PathBuf {
        d.path().join(name)
    }

    fn params(k: usize, lines: usize) -> SampleParams {
        SampleParams {
            samples_per_file: k,
            lines_per_sample: lines,
            seed: 7,
            max_bytes_per_sample: 4096,
            max_stream_bytes: 0,
        }
    }

    #[test]
    fn small_text_windows_exact() {
        let d = tmp();
        let path = p(&d, "a.txt");
        let mut f = File::create(&path).unwrap();
        for i in 0..200 {
            writeln!(f, "line {i}").unwrap();
        }
        drop(f);
        let out = sample_file(&path, Format::Text, Kind::Lines, Compression::None, &params(10, 50)).unwrap();
        // 200 lines / 50 per window = 4 windows; asking for 10 gives 4.
        assert_eq!(out.samples.len(), 4);
        let first = &out.samples[0];
        assert!(first.text.starts_with("line 0\n"));
        assert_eq!(first.text.lines().count(), 50);
        match first.locator {
            Locator::Bytes { byte_start, n_lines, .. } => {
                assert_eq!(byte_start, 0);
                assert_eq!(n_lines, 50);
            }
            _ => panic!("bad locator"),
        }
        // Locators must round-trip: reading the byte range gives the same lines.
        let data = std::fs::read(&path).unwrap();
        for s in &out.samples {
            if let Locator::Bytes { byte_start, byte_end, .. } = s.locator {
                let slice = &data[byte_start as usize..byte_end as usize];
                let txt = String::from_utf8_lossy(slice);
                assert_eq!(txt.trim_end(), s.text);
            }
        }
    }

    #[test]
    fn small_jsonl_rows() {
        let d = tmp();
        let path = p(&d, "a.jsonl");
        let mut f = File::create(&path).unwrap();
        for i in 0..1000 {
            writeln!(f, "{{\"id\": {i}, \"text\": \"row {i}\"}}").unwrap();
        }
        drop(f);
        let out = sample_file(&path, Format::Jsonl, Kind::Rows, Compression::None, &params(20, 50)).unwrap();
        assert_eq!(out.samples.len(), 20);
        for s in &out.samples {
            let v: serde_json::Value = serde_json::from_str(&s.text).unwrap();
            assert!(v["id"].is_number());
            assert_eq!(s.text.lines().count(), 1);
        }
        // Determinism
        let again = sample_file(&path, Format::Jsonl, Kind::Rows, Compression::None, &params(20, 50)).unwrap();
        let a: Vec<_> = out.samples.iter().map(|s| s.text.clone()).collect();
        let b: Vec<_> = again.samples.iter().map(|s| s.text.clone()).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn large_path_random_offsets() {
        let d = tmp();
        let path = p(&d, "big.jsonl");
        let mut f = std::io::BufWriter::new(File::create(&path).unwrap());
        for i in 0..5000 {
            writeln!(f, "{{\"id\": {i}, \"pad\": \"{}\"}}", "x".repeat(40)).unwrap();
        }
        drop(f);
        let mut rng = Rng::new(3);
        // Force the large-file path with a tiny small_limit.
        let out = sample_uncompressed(&path, 1, 25, &mut rng, 4096, false, 1).unwrap();
        assert_eq!(out.samples.len(), 25);
        let data = std::fs::read(&path).unwrap();
        let mut seen = std::collections::HashSet::new();
        for s in &out.samples {
            let v: serde_json::Value = serde_json::from_str(&s.text).expect("valid json line");
            assert!(seen.insert(v["id"].as_i64().unwrap()), "duplicate row");
            if let Locator::Bytes { byte_start, byte_end, n_lines } = s.locator {
                assert_eq!(n_lines, 1);
                let slice = &data[byte_start as usize..byte_end as usize];
                assert_eq!(String::from_utf8_lossy(slice).trim_end(), s.text);
            } else {
                panic!("bad locator");
            }
        }
        // Multi-line windows on the large path.
        let mut rng = Rng::new(3);
        let out = sample_uncompressed(&path, 10, 5, &mut rng, 1 << 20, false, 1).unwrap();
        assert_eq!(out.samples.len(), 5);
        for s in &out.samples {
            assert_eq!(s.text.lines().count(), 10);
        }
    }

    #[test]
    fn csv_header_prepended() {
        let d = tmp();
        let path = p(&d, "a.csv");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "id,text").unwrap();
        for i in 0..100 {
            writeln!(f, "{i},hello {i}").unwrap();
        }
        drop(f);
        let out = sample_file(&path, Format::Csv, Kind::Rows, Compression::None, &params(5, 50)).unwrap();
        assert_eq!(out.samples.len(), 5);
        for s in &out.samples {
            let mut lines = s.text.lines();
            assert_eq!(lines.next().unwrap(), "id,text");
            assert!(lines.next().unwrap().contains(",hello "));
            assert!(lines.next().is_none());
            if let Locator::Bytes { byte_start, .. } = s.locator {
                assert!(byte_start >= 8, "must not point at the header");
            }
        }
    }

    #[test]
    fn gzip_stream_reservoir() {
        let d = tmp();
        let path = p(&d, "a.jsonl.gz");
        let f = File::create(&path).unwrap();
        let mut enc = flate2::write::GzEncoder::new(f, flate2::Compression::fast());
        for i in 0..3000 {
            writeln!(enc, "{{\"id\": {i}}}").unwrap();
        }
        enc.finish().unwrap();
        let out = sample_file(&path, Format::Jsonl, Kind::Rows, Compression::Gzip, &params(30, 50)).unwrap();
        assert_eq!(out.samples.len(), 30);
        assert_eq!(out.row_count, Some(3000));
        let mut prev = -1;
        for s in &out.samples {
            let v: serde_json::Value = serde_json::from_str(&s.text).unwrap();
            let id = v["id"].as_i64().unwrap();
            match s.locator {
                Locator::StreamLines { line_start, n_lines } => {
                    assert_eq!(line_start, id);
                    assert_eq!(n_lines, 1);
                    assert!(line_start > prev);
                    prev = line_start;
                }
                _ => panic!("bad locator"),
            }
        }
    }

    #[test]
    fn zstd_stream_windows() {
        let d = tmp();
        let path = p(&d, "a.txt.zst");
        let f = File::create(&path).unwrap();
        let mut enc = zstd::stream::write::Encoder::new(f, 1).unwrap();
        for i in 0..1000 {
            writeln!(enc, "line {i}").unwrap();
        }
        enc.finish().unwrap();
        let out = sample_file(&path, Format::Text, Kind::Lines, Compression::Zstd, &params(5, 100)).unwrap();
        assert_eq!(out.samples.len(), 5);
        for s in &out.samples {
            assert_eq!(s.text.lines().count(), 100);
            if let Locator::StreamLines { line_start, .. } = s.locator {
                assert_eq!(line_start % 100, 0);
                assert!(s.text.starts_with(&format!("line {line_start}\n")));
            }
        }
    }

    #[test]
    fn parquet_row_selection() {
        use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::sync::Arc;

        let d = tmp();
        let path = p(&d, "a.parquet");
        let n = 10_000i64;
        let ids: Vec<i64> = (0..n).collect();
        let texts: Vec<String> = (0..n).map(|i| format!("document number {i}")).collect();
        let batch = RecordBatch::try_from_iter(vec![
            ("id", Arc::new(Int64Array::from(ids)) as ArrayRef),
            ("text", Arc::new(StringArray::from(texts)) as ArrayRef),
        ])
        .unwrap();
        let props = WriterProperties::builder().set_max_row_group_row_count(Some(1000)).build();
        let f = File::create(&path).unwrap();
        let mut w = ArrowWriter::try_new(f, batch.schema(), Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        let out = sample_file(&path, Format::Parquet, Kind::Rows, Compression::None, &params(40, 50)).unwrap();
        assert_eq!(out.row_count, Some(n));
        assert_eq!(out.samples.len(), 40);
        let mut prev = -1;
        for s in &out.samples {
            let v: serde_json::Value = serde_json::from_str(&s.text).unwrap();
            let id = v["id"].as_i64().unwrap();
            match s.locator {
                Locator::Row { index } => {
                    assert_eq!(index, id, "row index must match record");
                    assert!(index > prev);
                    prev = index;
                }
                _ => panic!("bad locator"),
            }
            assert_eq!(v["text"].as_str().unwrap(), format!("document number {id}"));
        }
    }

    #[test]
    fn json_sniff_routes() {
        let d = tmp();
        let nd = p(&d, "nd.json");
        std::fs::write(&nd, "{\"a\":1}\n{\"a\":2}\n{\"a\":3}\n").unwrap();
        let out = sample_file(&nd, Format::Json, Kind::Rows, Compression::None, &params(2, 50)).unwrap();
        assert_eq!(out.effective_format, Some(Format::Jsonl));
        assert_eq!(out.samples.len(), 2);

        let arr = p(&d, "arr.json");
        std::fs::write(&arr, "[\n{\"a\":1},\n{\"a\":2}\n]\n").unwrap();
        let out = sample_file(&arr, Format::Json, Kind::Rows, Compression::None, &params(2, 50)).unwrap();
        assert_eq!(out.effective_kind, Some(Kind::Lines));
        assert_eq!(out.samples.len(), 1);
    }

    #[test]
    fn truncation_cap() {
        let d = tmp();
        let path = p(&d, "long.txt");
        std::fs::write(&path, "é".repeat(5000)).unwrap();
        let mut prm = params(1, 50);
        prm.max_bytes_per_sample = 101; // odd cap: must not split a multibyte char
        let out = sample_file(&path, Format::Text, Kind::Lines, Compression::None, &prm).unwrap();
        assert_eq!(out.samples.len(), 1);
        assert!(out.samples[0].truncated);
        assert_eq!(out.samples[0].text.len(), 100);
    }
}
