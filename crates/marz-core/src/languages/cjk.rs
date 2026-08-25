//! Shared utilities for CJK language tokenizers.
//!
//! # Why bigrams, and only bigrams
//!
//! Chinese and Japanese are written without spaces, so `中文搜索引擎` is one
//! run of six characters that a reader parses as `中文 / 搜索 / 引擎`. Finding
//! those boundaries properly needs a dictionary (jieba, MeCab, Kuromoji) —
//! megabytes of data, and a hard dependency Marz deliberately does not take.
//!
//! The dictionary-free alternative, used by Lucene's `CJKAnalyzer` and
//! Elasticsearch's `cjk_bigram` filter, is to index every overlapping
//! character bigram:
//!
//! ```text
//! 中文搜索引擎  ->  中文 文搜 搜索 索引 引擎
//! ```
//!
//! A query for `搜索` finds the document because `搜索` is one of the indexed
//! bigrams. A query for a longer phrase like `搜索引擎` decomposes into
//! `搜索 索引 引擎`, all of which must be present — and, with positions, must
//! be *adjacent* — so precision stays high without a segmenter.
//!
//! ## The unigram mistake
//!
//! The previous implementation emitted unigrams *and* bigrams. That is a
//! serious error, not merely wasteful:
//!
//! - **Index size.** For a run of *n* characters it stores `2n-1` postings
//!   instead of `n-1`, roughly tripling the index once the term dictionary
//!   overhead is counted. Measured on real Chinese Wikipedia text: 1.34 tokens
//!   per source character, against ~1.0 for bigram-only.
//! - **Corrupted scoring.** BM25 divides by `field_length`. Doubling the token
//!   count doubles every field length, so the length-normalization denominator
//!   is wrong for every document, and the `avg_field_length` it is compared
//!   against is wrong too.
//! - **Terrible precision.** A single Han character is a morpheme, not a word.
//!   `的`, `是`, `中` appear in nearly every Chinese document, so unigram
//!   postings are enormous, near-zero-IDF lists that mostly add noise.
//!
//! Unigrams are emitted in exactly one case: a script run of length 1, where
//! there is no bigram to form. Dropping those would make a genuinely isolated
//! character unfindable.

use crate::normalize::normalize;
use crate::token::Token;
use crate::tokenizer::tokenize_with_separator;

/// Returns true for CJK Unified Ideographs (Han).
pub fn is_cjk_ideograph(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF        // CJK Unified Ideographs
            | 0x3400..=0x4DBF     // Extension A
            | 0xF900..=0xFAFF     // Compatibility Ideographs
            | 0x3005              // 々 iteration mark
            | 0x3007              // 〇 ideographic zero
            | 0x20000..=0x2A6DF   // Extension B
            | 0x2A700..=0x2B73F   // Extension C
            | 0x2B740..=0x2B81F   // Extension D
            | 0x2B820..=0x2CEAF   // Extension E
            | 0x2CEB0..=0x2EBEF   // Extension F
            | 0x30000..=0x3134F // Extension G
    )
}

/// Returns true for Hiragana.
pub fn is_hiragana(c: char) -> bool {
    matches!(c as u32, 0x3041..=0x309F)
}

/// Returns true for Katakana, including the prolonged sound mark.
pub fn is_katakana(c: char) -> bool {
    matches!(c as u32, 0x30A1..=0x30FF | 0x31F0..=0x31FF)
}

/// Returns true for Hangul syllables and Jamo.
pub fn is_hangul(c: char) -> bool {
    matches!(
        c as u32,
        0xAC00..=0xD7A3        // Hangul syllables
            | 0x1100..=0x11FF     // Jamo
            | 0x3130..=0x318F     // Compatibility Jamo
            | 0xA960..=0xA97F     // Jamo Extended-A
            | 0xD7B0..=0xD7FF // Jamo Extended-B
    )
}

/// Returns true for any CJK script character handled by this module.
pub fn is_cjk_char(c: char) -> bool {
    is_cjk_ideograph(c) || is_hiragana(c) || is_katakana(c) || is_hangul(c)
}

/// The script class of a character, used to segment mixed-script text.
///
/// Japanese switches script at morpheme boundaries — `検索エンジン` is Han then
/// Katakana — so treating a script change as a token boundary recovers real
/// word boundaries for free. Bigrams are never formed across a script change:
/// `索エ` spans two different words and is pure noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Script {
    /// Han ideographs.
    Han,
    /// Hiragana.
    Hiragana,
    /// Katakana.
    Katakana,
    /// Hangul.
    Hangul,
    /// Anything else: Latin, digits, punctuation, whitespace.
    Other,
}

/// Classify a character into a [`Script`].
pub fn script_of(c: char) -> Script {
    if is_cjk_ideograph(c) {
        Script::Han
    } else if is_hiragana(c) {
        Script::Hiragana
    } else if is_katakana(c) {
        Script::Katakana
    } else if is_hangul(c) {
        Script::Hangul
    } else {
        Script::Other
    }
}

/// Default separator characters for CJK languages.
pub const CJK_SEPARATORS: &str = " \t\n\r\x0C\x0B\x0D\u{00A0}";

/// Tokenize text into CJK bigrams plus separator-delimited non-CJK words.
///
/// `bigram_scripts` selects which scripts are bigrammed. Scripts not listed are
/// tokenized as whole runs — this is how Korean keeps its whitespace-delimited
/// words (see [`crate::languages::korean`]).
///
/// Token positions are `(char_offset, char_length)` into the *normalized* text,
/// so a bigram at offset 3 covers normalized characters 3 and 4. Adjacent
/// bigrams therefore have positions differing by exactly 1, which is what
/// phrase verification relies on.
pub fn tokenize_cjk(text: &str, bigram_scripts: &[Script]) -> Vec<Token> {
    let normalized = normalize(text);
    let chars: Vec<char> = normalized.chars().collect();
    let mut tokens: Vec<Token> = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        let script = script_of(chars[i]);

        if script == Script::Other {
            // Latin, digits, punctuation: delegate to the separator tokenizer,
            // then re-base its positions onto the full string.
            let run_start = i;
            while i < chars.len() && script_of(chars[i]) == Script::Other {
                i += 1;
            }
            let run: String = chars[run_start..i].iter().collect();
            for t in tokenize_with_separator(&run, CJK_SEPARATORS) {
                let (start, len) = t.position().unwrap_or((0, t.term.chars().count()));
                let index = tokens.len();
                tokens.push(Token::with_position(t.term, run_start + start, len, index));
            }
            continue;
        }

        // A run of a single CJK script.
        let run_start = i;
        while i < chars.len() && script_of(chars[i]) == script {
            i += 1;
        }
        let run_len = i - run_start;

        if !bigram_scripts.contains(&script) {
            // Not a bigrammed script: emit the whole run as one term.
            let term: String = chars[run_start..i].iter().collect();
            let index = tokens.len();
            tokens.push(Token::with_position(term, run_start, run_len, index));
            continue;
        }

        if run_len == 1 {
            // No bigram exists; emit the unigram so the character stays findable.
            let index = tokens.len();
            tokens.push(Token::with_position(
                chars[run_start].to_string(),
                run_start,
                1,
                index,
            ));
            continue;
        }

        for offset in 0..run_len - 1 {
            let bigram: String = chars[run_start + offset..run_start + offset + 2]
                .iter()
                .collect();
            let index = tokens.len();
            tokens.push(Token::with_position(bigram, run_start + offset, 2, index));
        }
    }

    tokens
}

/// Trimmer suitable for CJK languages: CJK terms are kept as-is, other terms
/// are stripped of surrounding non-word characters.
///
/// The trim is computed over character boundaries. An earlier byte-index
/// version (`s[start..end]` using `rfind(..).map(|i| i + 1)`) panicked on any
/// token whose last word character was multi-byte — which real Chinese,
/// Japanese and Korean text produces constantly via stray Greek letters,
/// accented Latin and the Japanese iteration mark `々`.
pub fn cjk_trim(token: &mut Token) -> bool {
    if token.term.chars().next().is_some_and(is_cjk_char) {
        return !token.term.is_empty();
    }
    token.update(|s| {
        let mut word_chars = s.char_indices().filter(|(_, c)| is_word_char(*c));
        match word_chars.next() {
            None => String::new(),
            Some(first) => {
                let last = word_chars.next_back().unwrap_or(first);
                s[first.0..last.0 + last.1.len_utf8()].to_string()
            }
        }
    });
    !token.term.is_empty()
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(tokens: &[Token]) -> Vec<String> {
        tokens.iter().map(|t| t.term.clone()).collect()
    }

    #[test]
    fn han_run_is_bigrams_only() {
        let tokens = tokenize_cjk("中文搜索引擎", &[Script::Han]);
        assert_eq!(
            terms(&tokens),
            ["中文", "文搜", "搜索", "索引", "引擎"],
            "6 chars must yield exactly 5 bigrams and no unigrams"
        );
    }

    #[test]
    fn bigram_positions_are_adjacent() {
        let tokens = tokenize_cjk("中文搜索", &[Script::Han]);
        let positions: Vec<_> = tokens.iter().map(|t| t.position().unwrap()).collect();
        assert_eq!(positions, [(0, 2), (1, 2), (2, 2)]);
    }

    #[test]
    fn isolated_char_becomes_a_unigram() {
        // Otherwise a lone character would be unindexable.
        let tokens = tokenize_cjk("我 said 好", &[Script::Han]);
        let t = terms(&tokens);
        assert!(t.contains(&"我".to_string()), "got {t:?}");
        assert!(t.contains(&"好".to_string()), "got {t:?}");
        assert!(t.contains(&"said".to_string()), "got {t:?}");
    }

    #[test]
    fn no_bigrams_across_script_boundaries() {
        // 検索 is Han, エンジン is Katakana. "索エ" would span two words.
        let tokens = tokenize_cjk("検索エンジン", &[Script::Han, Script::Katakana]);
        let t = terms(&tokens);
        assert_eq!(t, ["検索", "エン", "ンジ", "ジン"]);
        assert!(!t.contains(&"索エ".to_string()));
    }

    #[test]
    fn latin_inside_cjk_is_preserved() {
        let tokens = tokenize_cjk("使用 rust 编程", &[Script::Han]);
        let t = terms(&tokens);
        assert!(t.contains(&"rust".to_string()), "got {t:?}");
        assert!(t.contains(&"使用".to_string()), "got {t:?}");
        assert!(t.contains(&"编程".to_string()), "got {t:?}");
    }

    #[test]
    fn fullwidth_latin_folds_before_tokenizing() {
        let tokens = tokenize_cjk("ＲＵＳＴ言語", &[Script::Han]);
        assert!(terms(&tokens).contains(&"rust".to_string()));
    }

    #[test]
    fn non_bigram_script_emits_whole_runs() {
        // Korean: whitespace already delimits words, so do not bigram.
        let tokens = tokenize_cjk("검색 엔진", &[]);
        assert_eq!(terms(&tokens), ["검색", "엔진"]);
    }

    #[test]
    fn token_count_is_one_per_char_not_two() {
        // Regression guard for the unigram+bigram inflation bug.
        let text = "中文搜索引擎技术";
        let n = text.chars().count();
        let tokens = tokenize_cjk(text, &[Script::Han]);
        assert_eq!(tokens.len(), n - 1);
    }

    #[test]
    fn trim_does_not_panic_on_multibyte() {
        // Every one of these panicked in the byte-indexed version.
        for input in ["«café»", "”é”", "--é--", "δ", "々", "(π)"] {
            let mut t = Token::new(input);
            let _ = cjk_trim(&mut t);
        }
    }

    #[test]
    fn trim_strips_surrounding_punctuation() {
        let mut t = Token::new("(rust)");
        assert!(cjk_trim(&mut t));
        assert_eq!(t.term, "rust");
    }
}
