//! Built-in baseline classifier. Deliberately simple and transparent: it is a
//! pipeline smoke test and a "first opinion" shown in the review UI, not a
//! substitute for the model trained on your own annotations.

use super::{Backend, ClassifyItem, ClassifyResult};

#[derive(Default)]
pub struct Heuristic;

const STOPWORDS: &[&str] = &[
    "the", "and", "of", "to", "in", "is", "that", "it", "for", "was", "on", "are", "with", "as", "this", "be", "at",
    "by", "an", "from", "or", "have", "not", "but", "they", "which", "you", "were", "their", "has", "will", "can",
    "we", "his", "her", "there", "been", "if", "more", "when", "who", "would", "about", "also", "into", "than",
];

const CODE_MARKERS: &[&str] = &[
    "def ", "fn ", "import ", "#include", "class ", "return ", "public ", "private ", "static ", "void ", "const ",
    "let ", "var ", "function ", "=>", "->", "::", "();", "{}", "});", "if (", "for (", "while (", "elif ", "self.",
    "std::", "#!/", "</", "/>", "@Override", "SELECT ", "FROM ", "WHERE ", "print(", "println", "printf(",
];

const BOILERPLATE_MARKERS: &[&str] = &[
    "all rights reserved",
    "cookie",
    "privacy policy",
    "terms of service",
    "click here",
    "subscribe",
    "sign up",
    "log in",
    "share this",
    "read more",
    "advertisement",
    "javascript is disabled",
    "enable javascript",
    "skip to content",
    "follow us",
    "copyright ©",
    "lorem ipsum",
];

const MATH_MARKERS: &[&str] = &["\\frac", "\\sum", "\\int", "\\alpha", "\\beta", "\\theta", "$$", "\\begin{", "\\end{", "\\cdot", "\\sqrt", "\\mathbb", "\\left(", "\\right)"];

const AGENT_MARKERS: &[&str] = &[
    "\"role\": \"assistant\"",
    "\"role\": \"user\"",
    "\"role\":\"assistant\"",
    "\"tool_calls\"",
    "\"function_call\"",
    "<|im_start|>",
    "<|assistant|>",
    "Thought:",
    "Action:",
    "Observation:",
    "### Instruction",
    "### Response",
    "Human:",
    "Assistant:",
];

pub struct Features {
    pub len: usize,
    pub lines: usize,
    pub alpha_ratio: f64,
    pub digit_ratio: f64,
    pub non_ascii_ratio: f64,
    pub stopword_ratio: f64,
    pub code_hits: usize,
    pub boilerplate_hits: usize,
    pub math_hits: usize,
    pub agent_hits: usize,
    pub dup_line_ratio: f64,
    pub mean_line_len: f64,
    pub max_line_len: usize,
    pub symbol_ratio: f64,
    pub mean_word_len: f64,
}

pub fn features(text: &str) -> Features {
    let len = text.len();
    let mut alpha = 0usize;
    let mut digit = 0usize;
    let mut non_ascii = 0usize;
    let mut symbol = 0usize;
    for c in text.chars() {
        if !c.is_ascii() {
            non_ascii += 1;
        }
        if c.is_alphabetic() {
            alpha += 1;
        } else if c.is_ascii_digit() {
            digit += 1;
        } else if c.is_ascii_punctuation() {
            symbol += 1;
        }
    }
    let nchars = text.chars().count().max(1);
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric() && c != '\'').filter(|w| !w.is_empty()).collect();
    let nwords = words.len().max(1);
    let stop = words.iter().filter(|w| STOPWORDS.contains(w)).count();
    let mean_word_len = words.iter().map(|w| w.len()).sum::<usize>() as f64 / nwords as f64;
    let lines: Vec<&str> = text.lines().collect();
    let nlines = lines.len().max(1);
    let mut seen = std::collections::HashSet::new();
    let mut dups = 0usize;
    for l in &lines {
        let t = l.trim();
        if t.len() < 8 {
            continue;
        }
        if !seen.insert(t) {
            dups += 1;
        }
    }
    let max_line_len = lines.iter().map(|l| l.len()).max().unwrap_or(0);
    let mean_line_len = lines.iter().map(|l| l.len()).sum::<usize>() as f64 / nlines as f64;
    let count_hits = |markers: &[&str], hay: &str| markers.iter().filter(|m| hay.contains(&m.to_lowercase())).count();
    Features {
        len,
        lines: lines.len(),
        alpha_ratio: alpha as f64 / nchars as f64,
        digit_ratio: digit as f64 / nchars as f64,
        non_ascii_ratio: non_ascii as f64 / nchars as f64,
        stopword_ratio: stop as f64 / nwords as f64,
        code_hits: CODE_MARKERS.iter().filter(|m| text.contains(*m)).count(),
        boilerplate_hits: count_hits(BOILERPLATE_MARKERS, &lower),
        math_hits: MATH_MARKERS.iter().filter(|m| text.contains(*m)).count(),
        agent_hits: AGENT_MARKERS.iter().filter(|m| text.contains(*m)).count(),
        dup_line_ratio: dups as f64 / nlines as f64,
        mean_line_len,
        max_line_len,
        symbol_ratio: symbol as f64 / nchars as f64,
        mean_word_len,
    }
}

pub fn classify_one(item: &ClassifyItem) -> ClassifyResult {
    let f = features(&item.text);
    let mut tags: Vec<String> = Vec::new();
    let path_lower = item.path.to_lowercase();

    let is_code = f.code_hits >= 3 || (f.code_hits >= 1 && f.symbol_ratio > 0.12 && f.stopword_ratio < 0.06) || item.format == "code";
    if is_code {
        tags.push("code".into());
    }
    if f.agent_hits >= 2 || item.format == "jsonl" && f.agent_hits >= 1 {
        tags.push("agent-trace".into());
    }
    if f.math_hits >= 2 {
        tags.push("math".into());
    }
    if f.boilerplate_hits >= 2 || item.format == "html" {
        tags.push("web".into());
    }
    if f.boilerplate_hits >= 3 {
        tags.push("boilerplate".into());
    }
    if !is_code {
        if f.stopword_ratio >= 0.08 && f.non_ascii_ratio < 0.15 {
            tags.push("english".into());
        } else if f.non_ascii_ratio > 0.25 || (f.stopword_ratio < 0.02 && f.alpha_ratio > 0.5 && f.len > 200) {
            tags.push("non-english".into());
        }
    }
    if path_lower.contains("arxiv") || path_lower.contains("paper") || path_lower.contains("pubmed") || (f.math_hits >= 1 && f.stopword_ratio > 0.06) {
        tags.push("academic".into());
    }

    // Quality verdict.
    let mut bad_score: f64 = 0.0;
    let mut reasons: Vec<&str> = Vec::new();
    if f.len < 40 {
        bad_score += 0.6;
        reasons.push("very short");
    }
    if f.dup_line_ratio > 0.4 {
        bad_score += 0.5;
        reasons.push("repeated lines");
    }
    if !is_code && f.alpha_ratio < 0.45 && f.len >= 40 {
        bad_score += 0.4;
        reasons.push("low alphabetic ratio");
    }
    if f.boilerplate_hits >= 3 {
        bad_score += 0.4;
        reasons.push("boilerplate markers");
    }
    if !is_code && f.mean_word_len > 14.0 {
        bad_score += 0.3;
        reasons.push("unusual token length");
    }
    if !is_code && f.max_line_len > 20_000 {
        bad_score += 0.2;
        reasons.push("giant line");
    }
    if !is_code && f.symbol_ratio > 0.3 {
        bad_score += 0.4;
        reasons.push("symbol heavy");
    }
    let _ = reasons;
    let bad_score = bad_score.min(1.0);
    let (verdict, score) = if bad_score >= 0.5 {
        ("bad", 0.5 + bad_score / 2.0)
    } else {
        ("good", 1.0 - bad_score)
    };
    ClassifyResult {
        id: item.id,
        verdict: verdict.to_string(),
        score,
        tags,
    }
}

impl Backend for Heuristic {
    fn classify(&mut self, batch: &[ClassifyItem]) -> anyhow::Result<Vec<ClassifyResult>> {
        Ok(batch.iter().map(classify_one).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(text: &str, format: &str) -> ClassifyItem {
        ClassifyItem {
            id: 1,
            text: text.to_string(),
            path: "x".into(),
            format: format.into(),
        }
    }

    #[test]
    fn english_prose_is_good() {
        let r = classify_one(&item(
            "The committee reviewed the proposal and decided that it would be better to wait for more data before making a final decision on the matter.",
            "text",
        ));
        assert_eq!(r.verdict, "good");
        assert!(r.tags.contains(&"english".to_string()));
    }

    #[test]
    fn code_is_tagged() {
        let r = classify_one(&item(
            "fn main() {\n    let x = 5;\n    println!(\"{}\", x);\n    return;\n}\n",
            "code",
        ));
        assert!(r.tags.contains(&"code".to_string()));
    }

    #[test]
    fn junk_is_bad() {
        let r = classify_one(&item("###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ######", "text"));
        assert_eq!(r.verdict, "bad");
        let r = classify_one(&item("hi", "text"));
        assert_eq!(r.verdict, "bad");
    }
}
