//! Built-in trainable classifier: logistic regression over hashed word and
//! character n-grams plus a handful of dense text statistics. Small, fast,
//! dependency-free, and good enough to bootstrap from a few hundred human
//! decisions. Trained by `train.rs`, served in-process by `BuiltinBackend`.
//!
//! Model file layout (`.gccfm`):
//!   magic "GCCFM1" | u32 header_len | header JSON | idf f32[n_features]
//!   | verdict f32[n_features + 1] | tags[i] f32[n_features + 1] ...
//! The trailing weight of each head is its bias.

use super::{heuristic, Backend, ClassifyItem, ClassifyResult};
use anyhow::{anyhow, Context};
use serde::{Deserialize, Serialize};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;

pub const MAGIC: &[u8; 6] = b"GCCFM1";
/// Reserved low indices for dense statistics; hashed features start after.
pub const DENSE: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    pub dim_bits: u32,
    /// Characters of text considered for word features.
    pub word_chars: usize,
    /// Characters of text considered for char n-grams (costlier).
    pub char_chars: usize,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        FeatureConfig {
            dim_bits: 18,
            word_chars: 10_000,
            char_chars: 3_000,
        }
    }
}

impl FeatureConfig {
    pub fn n_features(&self) -> usize {
        DENSE + (1usize << self.dim_bits)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHeader {
    pub name: String,
    pub created_at: String,
    pub features: FeatureConfig,
    pub tag_names: Vec<String>,
    pub tag_threshold: f32,
    pub train_rows: usize,
    pub metrics: serde_json::Value,
}

pub struct BuiltinModel {
    pub header: ModelHeader,
    pub idf: Vec<f32>,
    pub verdict: Vec<f32>,
    pub tags: Vec<Vec<f32>>,
}

/// Sparse feature vector: sorted, deduplicated (index, value) pairs.
pub type Sparse = Vec<(u32, f32)>;

#[inline]
fn fnv1a(ns: u8, bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ (ns as u64);
    h = h.wrapping_mul(0x0000_0100_0000_01b3);
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn take_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// Raw term counts (sublinear tf applied) + dense statistics. No idf, no
/// normalisation — those depend on the training corpus.
pub fn raw_features(text: &str, cfg: &FeatureConfig) -> Sparse {
    let mask = (1u64 << cfg.dim_bits) - 1;
    let idx = |ns: u8, bytes: &[u8]| -> u32 { (DENSE as u64 + (fnv1a(ns, bytes) & mask)) as u32 };
    let mut out: Sparse = Vec::with_capacity(4096);

    // Dense stats (each roughly in [0, 1]).
    let f = heuristic::features(text);
    let clamp = |v: f64| v.clamp(0.0, 1.0) as f32;
    let dense = [
        clamp((1.0 + f.len as f64).ln() / 12.0),
        clamp((1.0 + f.lines as f64).ln() / 8.0),
        clamp(f.alpha_ratio),
        clamp(f.digit_ratio),
        clamp(f.non_ascii_ratio),
        clamp(f.symbol_ratio),
        clamp(f.stopword_ratio * 5.0),
        clamp(f.code_hits as f64 / 10.0),
        clamp(f.boilerplate_hits as f64 / 5.0),
        clamp(f.math_hits as f64 / 5.0),
        clamp(f.agent_hits as f64 / 5.0),
        clamp(f.dup_line_ratio),
        clamp((1.0 + f.mean_line_len).ln() / 8.0),
        clamp((1.0 + f.max_line_len as f64).ln() / 12.0),
        clamp(f.mean_word_len / 15.0),
        clamp(text.chars().filter(|c| c.is_whitespace()).count() as f64 / text.chars().count().max(1) as f64),
    ];
    for (i, v) in dense.iter().enumerate() {
        if *v != 0.0 {
            out.push((i as u32, *v));
        }
    }

    // Word unigrams + bigrams.
    let lower = take_chars(text, cfg.word_chars).to_lowercase();
    let mut prev: Option<&str> = None;
    for tok in lower.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '\'')) {
        if tok.is_empty() || tok.len() > 40 {
            prev = None;
            continue;
        }
        out.push((idx(b'w', tok.as_bytes()), 1.0));
        if let Some(p) = prev {
            let mut key = Vec::with_capacity(p.len() + tok.len() + 1);
            key.extend_from_slice(p.as_bytes());
            key.push(b' ');
            key.extend_from_slice(tok.as_bytes());
            out.push((idx(b'b', &key), 1.0));
        }
        prev = Some(tok);
    }

    // Character 3- and 4-grams within padded tokens (char_wb style).
    let char_src = take_chars(&lower, cfg.char_chars);
    let mut buf: Vec<char> = Vec::with_capacity(48);
    let mut key = String::with_capacity(8);
    for tok in char_src.split(|c: char| c.is_whitespace()) {
        if tok.is_empty() || tok.len() > 40 {
            continue;
        }
        buf.clear();
        buf.push(' ');
        buf.extend(tok.chars());
        buf.push(' ');
        for n in 3..=4usize {
            if buf.len() < n {
                continue;
            }
            for w in buf.windows(n) {
                key.clear();
                key.extend(w.iter());
                out.push((idx(b'c', key.as_bytes()), 1.0));
            }
        }
    }

    // Merge duplicates, sublinear tf.
    out.sort_unstable_by_key(|(i, _)| *i);
    let mut merged: Sparse = Vec::with_capacity(out.len() / 2);
    for (i, v) in out {
        match merged.last_mut() {
            Some((li, lv)) if *li == i => *lv += v,
            _ => merged.push((i, v)),
        }
    }
    for (i, v) in merged.iter_mut() {
        if (*i as usize) >= DENSE {
            *v = 1.0 + v.ln();
        }
    }
    merged
}

/// Apply idf to hashed features and L2-normalise them (dense stats untouched).
pub fn finish_features(x: &mut Sparse, idf: &[f32]) {
    let mut norm = 0.0f32;
    for (i, v) in x.iter_mut() {
        if (*i as usize) >= DENSE {
            *v *= idf[*i as usize];
            norm += *v * *v;
        }
    }
    if norm > 0.0 {
        let inv = 1.0 / norm.sqrt();
        for (i, v) in x.iter_mut() {
            if (*i as usize) >= DENSE {
                *v *= inv;
            }
        }
    }
}

#[inline]
pub fn dot(w: &[f32], x: &[(u32, f32)]) -> f32 {
    let mut s = w[w.len() - 1];
    for &(i, v) in x {
        s += w[i as usize] * v;
    }
    s
}

#[inline]
pub fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

impl BuiltinModel {
    pub fn featurize(&self, text: &str) -> Sparse {
        let mut x = raw_features(text, &self.header.features);
        finish_features(&mut x, &self.idf);
        x
    }

    pub fn predict_features(&self, x: &Sparse) -> (String, f64, Vec<String>) {
        let p_good = sigmoid(dot(&self.verdict, x));
        let (verdict, score) = if p_good >= 0.5 { ("good", p_good) } else { ("bad", 1.0 - p_good) };
        let mut tags = Vec::new();
        for (w, name) in self.tags.iter().zip(&self.header.tag_names) {
            if sigmoid(dot(w, x)) >= self.header.tag_threshold {
                tags.push(name.clone());
            }
        }
        (verdict.to_string(), score as f64, tags)
    }

    pub fn predict(&self, text: &str) -> (String, f64, Vec<String>) {
        let x = self.featurize(text);
        self.predict_features(&x)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut w = BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
        let header = serde_json::to_vec(&self.header)?;
        w.write_all(MAGIC)?;
        w.write_all(&(header.len() as u32).to_le_bytes())?;
        w.write_all(&header)?;
        let n = self.header.features.n_features();
        if self.idf.len() != n || self.verdict.len() != n + 1 || self.tags.iter().any(|t| t.len() != n + 1) {
            return Err(anyhow!("model vectors do not match feature config"));
        }
        for v in std::iter::once(&self.idf).chain(std::iter::once(&self.verdict)).chain(self.tags.iter()) {
            let mut bytes = Vec::with_capacity(v.len() * 4);
            for f in v {
                bytes.extend_from_slice(&f.to_le_bytes());
            }
            w.write_all(&bytes)?;
        }
        w.flush()?;
        Ok(())
    }

    pub fn load(path: &Path) -> anyhow::Result<BuiltinModel> {
        let mut r = BufReader::with_capacity(1 << 20, std::fs::File::open(path).with_context(|| format!("opening model {}", path.display()))?);
        let mut magic = [0u8; 6];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(anyhow!("{} is not a GCCF built-in model", path.display()));
        }
        let mut len = [0u8; 4];
        r.read_exact(&mut len)?;
        let mut header = vec![0u8; u32::from_le_bytes(len) as usize];
        r.read_exact(&mut header)?;
        let header: ModelHeader = serde_json::from_slice(&header)?;
        let n = header.features.n_features();
        let read_vec = |r: &mut BufReader<std::fs::File>, len: usize| -> anyhow::Result<Vec<f32>> {
            let mut bytes = vec![0u8; len * 4];
            r.read_exact(&mut bytes)?;
            Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        };
        let idf = read_vec(&mut r, n)?;
        let verdict = read_vec(&mut r, n + 1)?;
        let mut tags = Vec::with_capacity(header.tag_names.len());
        for _ in 0..header.tag_names.len() {
            tags.push(read_vec(&mut r, n + 1)?);
        }
        Ok(BuiltinModel {
            header,
            idf,
            verdict,
            tags,
        })
    }
}

pub struct BuiltinBackend {
    pub model: Arc<BuiltinModel>,
}

impl Backend for BuiltinBackend {
    fn classify(&mut self, batch: &[ClassifyItem]) -> anyhow::Result<Vec<ClassifyResult>> {
        Ok(batch
            .iter()
            .map(|item| {
                let (verdict, score, tags) = self.model.predict(&item.text);
                ClassifyResult {
                    id: item.id,
                    verdict,
                    score,
                    tags,
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_are_deterministic_and_bounded() {
        let cfg = FeatureConfig::default();
        let a = raw_features("The quick brown fox jumps over the lazy dog.", &cfg);
        let b = raw_features("The quick brown fox jumps over the lazy dog.", &cfg);
        assert_eq!(a, b);
        assert!(a.windows(2).all(|w| w[0].0 < w[1].0), "sorted and deduped");
        assert!(a.iter().all(|(i, _)| (*i as usize) < cfg.n_features()));
        assert!(a.len() > 20);
    }

    #[test]
    fn save_load_round_trip() {
        let cfg = FeatureConfig { dim_bits: 10, ..Default::default() };
        let n = cfg.n_features();
        let model = BuiltinModel {
            header: ModelHeader {
                name: "t".into(),
                created_at: "now".into(),
                features: cfg,
                tag_names: vec!["code".into()],
                tag_threshold: 0.5,
                train_rows: 3,
                metrics: serde_json::json!({}),
            },
            idf: vec![1.0; n],
            verdict: (0..=n).map(|i| i as f32 * 0.001).collect(),
            tags: vec![vec![0.5; n + 1]],
        };
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.gccfm");
        model.save(&p).unwrap();
        let back = BuiltinModel::load(&p).unwrap();
        assert_eq!(back.verdict, model.verdict);
        assert_eq!(back.tags, model.tags);
        assert_eq!(back.header.tag_names, vec!["code".to_string()]);
        let (v, s, t) = back.predict("fn main() {}");
        assert!(v == "good" || v == "bad");
        assert!(s >= 0.5 && s <= 1.0);
        assert_eq!(t, vec!["code".to_string()]);
    }
}
