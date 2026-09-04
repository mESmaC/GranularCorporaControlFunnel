//! File format classification. Everything GCCF knows about a file before it
//! opens it comes from here; sniffing (opening the file) is done lazily at
//! sampling time for ambiguous cases.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Parquet,
    Jsonl,
    Csv,
    Tsv,
    Json,
    Text,
    Markdown,
    Code,
    Html,
    Xml,
    Unknown,
    Binary,
}

impl Format {
    pub fn as_str(&self) -> &'static str {
        match self {
            Format::Parquet => "parquet",
            Format::Jsonl => "jsonl",
            Format::Csv => "csv",
            Format::Tsv => "tsv",
            Format::Json => "json",
            Format::Text => "text",
            Format::Markdown => "markdown",
            Format::Code => "code",
            Format::Html => "html",
            Format::Xml => "xml",
            Format::Unknown => "unknown",
            Format::Binary => "binary",
        }
    }

    pub fn parse(s: &str) -> Format {
        match s {
            "parquet" => Format::Parquet,
            "jsonl" => Format::Jsonl,
            "csv" => Format::Csv,
            "tsv" => Format::Tsv,
            "json" => Format::Json,
            "text" => Format::Text,
            "markdown" => Format::Markdown,
            "code" => Format::Code,
            "html" => Format::Html,
            "xml" => Format::Xml,
            "binary" => Format::Binary,
            _ => Format::Unknown,
        }
    }

    pub fn all() -> &'static [Format] {
        &[
            Format::Parquet,
            Format::Jsonl,
            Format::Csv,
            Format::Tsv,
            Format::Json,
            Format::Text,
            Format::Markdown,
            Format::Code,
            Format::Html,
            Format::Xml,
            Format::Unknown,
            Format::Binary,
        ]
    }
}

/// How samples are cut from the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// One sample == one record (parquet row, jsonl line, csv row).
    Rows,
    /// One sample == N consecutive lines of text.
    Lines,
    /// Cannot be sampled (binary, unsupported compression, ...).
    Unsupported,
}

impl Kind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::Rows => "rows",
            Kind::Lines => "lines",
            Kind::Unsupported => "unsupported",
        }
    }
    pub fn parse(s: &str) -> Kind {
        match s {
            "rows" => Kind::Rows,
            "lines" => Kind::Lines,
            _ => Kind::Unsupported,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Compression {
    None,
    Gzip,
    Zstd,
    Unsupported,
}

impl Compression {
    pub fn as_str(&self) -> &'static str {
        match self {
            Compression::None => "none",
            Compression::Gzip => "gzip",
            Compression::Zstd => "zstd",
            Compression::Unsupported => "unsupported",
        }
    }
    pub fn parse(s: &str) -> Compression {
        match s {
            "none" => Compression::None,
            "gzip" => Compression::Gzip,
            "zstd" => Compression::Zstd,
            _ => Compression::Unsupported,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Classification {
    pub format: Format,
    pub kind: Kind,
    pub compression: Compression,
}

const CODE_EXTS: &[&str] = &[
    "py", "pyi", "rs", "js", "mjs", "cjs", "ts", "tsx", "jsx", "c", "h", "cpp", "cc", "cxx", "hpp",
    "hh", "hxx", "java", "go", "rb", "php", "cs", "swift", "kt", "kts", "scala", "sh", "bash",
    "zsh", "fish", "ps1", "psm1", "bat", "cmd", "sql", "lua", "r", "jl", "hs", "ml", "mli", "ex",
    "exs", "erl", "clj", "cljs", "dart", "zig", "nim", "v", "sv", "svh", "vhd", "vhdl", "asm", "s",
    "cu", "cuh", "cl", "toml", "yaml", "yml", "ini", "cfg", "conf", "cmake", "make", "mk", "proto",
    "graphql", "gql", "css", "scss", "less", "vue", "svelte", "tf", "hcl", "nix", "dockerfile",
    "gradle", "groovy", "m", "mm", "pl", "pm", "tcl", "f", "f90", "f95", "for", "pas", "ada",
    "adb", "ads", "lisp", "el", "scm", "rkt", "coffee", "elm", "purs", "re", "res", "sol", "wat",
    "wasm.txt", "ipynb",
];

const TEXT_EXTS: &[&str] = &["txt", "text", "log", "rst", "tex", "bib", "srt", "vtt", "csvtxt", "wiki", "org", "adoc", "asciidoc"];
const MD_EXTS: &[&str] = &["md", "markdown", "mdx", "mdown"];
const HTML_EXTS: &[&str] = &["html", "htm", "xhtml"];
const XML_EXTS: &[&str] = &["xml", "xsl", "xsd", "svg", "rss", "atom"];
const BINARY_EXTS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff", "tif", "ico", "pdf", "doc", "docx", "xls",
    "xlsx", "ppt", "pptx", "zip", "tar", "7z", "rar", "bz2", "xz", "lz4", "exe", "dll", "so",
    "dylib", "bin", "dat", "pt", "pth", "ckpt", "safetensors", "npy", "npz", "pkl", "pickle",
    "arrow", "feather", "orc", "avro", "db", "sqlite", "sqlite3", "wav", "mp3", "flac", "ogg",
    "mp4", "mkv", "mov", "avi", "webm", "iso", "img", "dmg", "class", "jar", "o", "a", "lib",
    "pyc", "pyo", "wasm", "ttf", "otf", "woff", "woff2", "eot", "idx", "index", "lock",
];

fn lower_ext(name: &str) -> Option<String> {
    let idx = name.rfind('.')?;
    if idx == 0 || idx + 1 >= name.len() {
        return None;
    }
    Some(name[idx + 1..].to_ascii_lowercase())
}

/// Classify a file purely from its name. Returns the classification and the
/// "logical" extension (with compression suffix stripped) for display.
pub fn classify_name(file_name: &str) -> (Classification, String) {
    let mut name = file_name.to_string();
    let mut compression = Compression::None;

    if let Some(ext) = lower_ext(&name) {
        match ext.as_str() {
            "gz" | "gzip" => {
                compression = Compression::Gzip;
                name.truncate(name.len() - ext.len() - 1);
            }
            "zst" | "zstd" => {
                compression = Compression::Zstd;
                name.truncate(name.len() - ext.len() - 1);
            }
            "bz2" | "xz" | "lz4" | "lzma" | "br" => {
                // Not supported for sampling: mark as unsupported compression but
                // still classify the inner format for reporting.
                compression = Compression::Unsupported;
                name.truncate(name.len() - ext.len() - 1);
            }
            _ => {}
        }
    }

    let ext = lower_ext(&name).unwrap_or_default();
    let lower_name = name.to_ascii_lowercase();

    let format = match ext.as_str() {
        "parquet" | "pq" => Format::Parquet,
        "jsonl" | "ndjson" | "jsonlines" | "jsonl.txt" => Format::Jsonl,
        "json" => Format::Json,
        "csv" => Format::Csv,
        "tsv" | "tab" => Format::Tsv,
        e if MD_EXTS.contains(&e) => Format::Markdown,
        e if HTML_EXTS.contains(&e) => Format::Html,
        e if XML_EXTS.contains(&e) => Format::Xml,
        e if TEXT_EXTS.contains(&e) => Format::Text,
        e if CODE_EXTS.contains(&e) => Format::Code,
        e if BINARY_EXTS.contains(&e) => Format::Binary,
        "" => {
            // Extensionless: a few well-known code/text names, otherwise unknown
            // (sniffed at scan time by the caller if it wants to).
            if matches!(
                lower_name.as_str(),
                "dockerfile" | "makefile" | "cmakelists" | "license" | "readme" | "changelog" | "notice" | "authors" | "copying"
            ) {
                Format::Code
            } else {
                Format::Unknown
            }
        }
        _ => Format::Unknown,
    };

    let kind = kind_for(format, compression);
    (
        Classification {
            format,
            kind,
            compression,
        },
        ext,
    )
}

pub fn kind_for(format: Format, compression: Compression) -> Kind {
    if compression == Compression::Unsupported {
        return Kind::Unsupported;
    }
    match format {
        Format::Parquet => {
            if compression == Compression::None {
                Kind::Rows
            } else {
                // Parquet inside gzip/zstd cannot be random-accessed; skip.
                Kind::Unsupported
            }
        }
        Format::Jsonl | Format::Csv | Format::Tsv | Format::Json => Kind::Rows,
        Format::Text | Format::Markdown | Format::Code | Format::Html | Format::Xml => Kind::Lines,
        Format::Unknown => Kind::Lines,
        Format::Binary => Kind::Unsupported,
    }
}

/// Look at the first bytes of a file to decide whether it is text.
/// Returns `Some(is_text)`; `None` if the file could not be read.
pub fn sniff_is_text(path: &Path) -> Option<bool> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut buf = [0u8; 2048];
    let n = f.read(&mut buf).ok()?;
    if n == 0 {
        return Some(true);
    }
    let slice = &buf[..n];
    if slice.contains(&0u8) {
        return Some(false);
    }
    // Heuristic: too many control chars => binary.
    let ctrl = slice
        .iter()
        .filter(|&&b| b < 0x20 && !matches!(b, b'\n' | b'\r' | b'\t' | 0x0c))
        .count();
    Some(ctrl * 20 < n)
}

/// Sniff whether a `.json` file is newline-delimited objects (jsonl-like) or a
/// single document (array / pretty printed object).
pub fn sniff_json_is_ndjson(head: &[u8]) -> bool {
    let trimmed = head
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .map(|i| &head[i..])
        .unwrap_or(&[]);
    if trimmed.is_empty() {
        return false;
    }
    if trimmed[0] != b'{' {
        return false;
    }
    // First line must close the object => `}` right before the first newline.
    match memchr::memchr(b'\n', trimmed) {
        Some(nl) => {
            let line = &trimmed[..nl];
            let line = line.iter().rev().position(|b| !b.is_ascii_whitespace()).map(|i| &line[..line.len() - i]).unwrap_or(line);
            line.ends_with(b"}")
        }
        None => trimmed.ends_with(b"}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_basic() {
        let (c, _) = classify_name("shard-0001.jsonl.zst");
        assert_eq!(c.format, Format::Jsonl);
        assert_eq!(c.kind, Kind::Rows);
        assert_eq!(c.compression, Compression::Zstd);

        let (c, _) = classify_name("data.parquet");
        assert_eq!(c.format, Format::Parquet);
        assert_eq!(c.kind, Kind::Rows);

        let (c, _) = classify_name("data.parquet.gz");
        assert_eq!(c.kind, Kind::Unsupported);

        let (c, _) = classify_name("main.rs");
        assert_eq!(c.format, Format::Code);
        assert_eq!(c.kind, Kind::Lines);

        let (c, _) = classify_name("photo.JPG");
        assert_eq!(c.format, Format::Binary);
        assert_eq!(c.kind, Kind::Unsupported);

        let (c, _) = classify_name("notes.txt.xz");
        assert_eq!(c.compression, Compression::Unsupported);
        assert_eq!(c.kind, Kind::Unsupported);

        let (c, _) = classify_name("Dockerfile");
        assert_eq!(c.format, Format::Code);
    }

    #[test]
    fn ndjson_sniff() {
        assert!(sniff_json_is_ndjson(b"{\"a\":1}\n{\"a\":2}\n"));
        assert!(!sniff_json_is_ndjson(b"[\n{\"a\":1},\n{\"a\":2}\n]"));
        assert!(!sniff_json_is_ndjson(b"{\n  \"a\": 1\n}\n"));
    }
}
