//! Deterministic script-based router: picks "english" vs "multilingual" checkpoint from the
//! state text before running inference — never from model confidence (see the project README's
//! Khmer example: an English-only checkpoint can be confidently wrong on unreadable scripts).
//!
//! Two layers: a fast "22-alphabet" Unicode script detector handles the common, cheap case
//! (obviously non-Latin script -> multilingual, no need to run a detector model at all); for
//! Latin-script text — where script alone can't tell English from French, Vietnamese, etc. — this
//! defers to [lingua](https://github.com/pemistahl/lingua-rs), a real statistical language-ID
//! library, instead of a hand-rolled stopword heuristic (an earlier version of this router used
//! one and it misrouted ordinary English — see `english_stays_english` below).

use std::sync::OnceLock;

use lingua::{Language, LanguageDetector, LanguageDetectorBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Checkpoint {
    English,
    Multilingual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Script {
    Latin,
    Cyrillic,
    Greek,
    Armenian,
    Hebrew,
    Arabic,
    Devanagari,
    Bengali,
    Gurmukhi,
    Gujarati,
    Oriya,
    Tamil,
    Telugu,
    Kannada,
    Malayalam,
    Sinhala,
    Thai,
    Lao,
    Georgian,
    Hangul,
    Ethiopic,
    Khmer,
    Myanmar,
    Han, // CJK ideographs (Chinese; also used by Japanese Kanji)
    Hiragana,
    Katakana,
}

const SCRIPT_RANGES: &[(Script, u32, u32)] = &[
    (Script::Latin, 0x0041, 0x024F),
    (Script::Greek, 0x0370, 0x03FF),
    (Script::Cyrillic, 0x0400, 0x04FF),
    (Script::Armenian, 0x0530, 0x058F),
    (Script::Hebrew, 0x0590, 0x05FF),
    (Script::Arabic, 0x0600, 0x06FF),
    (Script::Devanagari, 0x0900, 0x097F),
    (Script::Bengali, 0x0980, 0x09FF),
    (Script::Gurmukhi, 0x0A00, 0x0A7F),
    (Script::Gujarati, 0x0A80, 0x0AFF),
    (Script::Oriya, 0x0B00, 0x0B7F),
    (Script::Tamil, 0x0B80, 0x0BFF),
    (Script::Telugu, 0x0C00, 0x0C7F),
    (Script::Kannada, 0x0C80, 0x0CFF),
    (Script::Malayalam, 0x0D00, 0x0D7F),
    (Script::Sinhala, 0x0D80, 0x0DFF),
    (Script::Thai, 0x0E00, 0x0E7F),
    (Script::Lao, 0x0E80, 0x0EFF),
    (Script::Georgian, 0x10A0, 0x10FF),
    (Script::Hangul, 0xAC00, 0xD7A3),
    (Script::Ethiopic, 0x1200, 0x137F),
    (Script::Khmer, 0x1780, 0x17FF),
    (Script::Myanmar, 0x1000, 0x109F),
    (Script::Hiragana, 0x3040, 0x309F),
    (Script::Katakana, 0x30A0, 0x30FF),
    (Script::Han, 0x4E00, 0x9FFF),
];

fn classify_char(ch: char) -> Option<Script> {
    let cp = ch as u32;
    SCRIPT_RANGES.iter().find(|&&(_, lo, hi)| cp >= lo && cp <= hi).map(|&(s, _, _)| s)
}

/// Per-script counts over a text: how many codepoints fall in each detected script's block.
pub fn script_histogram(text: &str) -> Vec<(Script, usize)> {
    let mut counts: std::collections::HashMap<Script, usize> = std::collections::HashMap::new();
    for ch in text.chars() {
        if let Some(s) = classify_char(ch) {
            *counts.entry(s).or_insert(0) += 1;
        }
    }
    let mut v: Vec<_> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    v
}

fn detector() -> &'static LanguageDetector {
    static DETECTOR: OnceLock<LanguageDetector> = OnceLock::new();
    DETECTOR.get_or_init(|| LanguageDetectorBuilder::from_all_languages().build())
}

/// Route a state's text to the checkpoint that can actually read it.
///
/// Dominant non-Latin script -> multilingual immediately (cheap, and catches the README's
/// motivating case: Khmer scoring 0% accuracy at 95% confidence on the English checkpoint — a
/// script it can't read at all, where model confidence gives no warning). Dominant Latin script
/// (or no alphabetic content) -> run `lingua` to tell English apart from other Latin-script
/// languages (French, Vietnamese, Indonesian, ...), which needs an actual language-ID model, not
/// a Unicode block check — a hand-rolled stopword-based version of this tried, and misrouted
/// ordinary English (see `english_stays_english` below) because its word list was too small.
pub fn route(text: &str) -> Checkpoint {
    let hist = script_histogram(text);
    match hist.first().map(|&(s, _)| s) {
        Some(s) if s != Script::Latin => return Checkpoint::Multilingual,
        _ => {}
    }
    match detector().detect_language_of(text) {
        Some(Language::English) | None => Checkpoint::English,
        Some(_) => Checkpoint::Multilingual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_stays_english() {
        let t = "A customer says a password reset succeeded, but every login attempt still returns 'account locked'.";
        assert_eq!(route(t), Checkpoint::English);
    }

    #[test]
    fn devanagari_routes_multilingual() {
        assert_eq!(route("मुझसे दो बार शुल्क लिया गया"), Checkpoint::Multilingual);
    }
}
