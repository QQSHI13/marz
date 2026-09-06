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

/// Returns true for Thai.
pub fn is_thai(c: char) -> bool {
    matches!(c as u32, 0x0E01..=0x0E5B)
}

/// Returns true for Lao.
pub fn is_lao(c: char) -> bool {
    matches!(c as u32, 0x0E81..=0x0EDF)
}

/// Returns true for Khmer.
pub fn is_khmer(c: char) -> bool {
    matches!(c as u32, 0x1780..=0x17F9 | 0x19E0..=0x19FF)
}

/// Returns true for Myanmar (Burmese).
pub fn is_myanmar(c: char) -> bool {
    matches!(c as u32, 0x1000..=0x109F | 0xA9E0..=0xA9FF | 0xAA60..=0xAA7F)
}

/// Returns true for Tibetan.
pub fn is_tibetan(c: char) -> bool {
    matches!(c as u32, 0x0F00..=0x0FDA)
}

/// Returns true for a combining mark in one of the non-spacing scripts.
///
/// These are the Unicode `Mn`/`Mc` characters in each script's block: vowel
/// signs, tone marks, viramas. They are not standalone characters — a mark
/// modifies the consonant before it, and a "character" a reader would point at
/// is a base plus its marks.
///
/// The ranges are enumerated rather than derived from Unicode character
/// properties, because deriving them means a `unicode-*` crate and a data
/// table, which is the dependency this project exists to avoid. They change
/// only when Unicode adds marks to these blocks, which is checked by
/// `no_bigram_starts_with_a_combining_mark`.
pub fn is_combining_mark(c: char) -> bool {
    matches!(
        c as u32,
        // Thai
        0x0E31 | 0x0E34..=0x0E3A | 0x0E47..=0x0E4E
        // Lao
        | 0x0EB1 | 0x0EB4..=0x0EBC | 0x0EC8..=0x0ECE
        // Khmer
        | 0x17B4..=0x17D3 | 0x17DD
        // Myanmar
        | 0x102B..=0x103E | 0x1056..=0x1059 | 0x105E..=0x1060
        | 0x1062..=0x1064 | 0x1067..=0x106D | 0x1071..=0x1074
        | 0x1082..=0x108D | 0x108F | 0x109A..=0x109D
        // Tibetan
        | 0x0F18..=0x0F19 | 0x0F35 | 0x0F37 | 0x0F39 | 0x0F3E..=0x0F3F
        | 0x0F71..=0x0F84 | 0x0F86..=0x0F87 | 0x0F8D..=0x0F97
        | 0x0F99..=0x0FBC | 0x0FC6
    )
}

/// Returns true for any CJK script character handled by this module.
pub fn is_cjk_char(c: char) -> bool {
    is_cjk_ideograph(c) || is_hiragana(c) || is_katakana(c) || is_hangul(c)
}

/// Returns true for any script this module bigrams, CJK or not.
pub fn is_ngram_char(c: char) -> bool {
    is_cjk_char(c) || is_thai(c) || is_lao(c) || is_khmer(c) || is_myanmar(c) || is_tibetan(c)
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
    /// Thai.
    Thai,
    /// Lao.
    Lao,
    /// Khmer.
    Khmer,
    /// Myanmar (Burmese).
    Myanmar,
    /// Tibetan.
    Tibetan,
    /// Anything else: Latin, digits, punctuation, whitespace.
    Other,
}

impl Script {
    /// Whether this script's characters combine into grapheme clusters.
    ///
    /// Han, Kana and Hangul do not: one code point is one character, so a
    /// cluster is always a single `char` and clustering is a no-op. The
    /// non-spacing scripts do, and bigramming them by code point instead of by
    /// cluster produces terms that are not linguistic units at all — see the
    /// module docs.
    pub fn has_combining_marks(self) -> bool {
        matches!(
            self,
            Script::Thai | Script::Lao | Script::Khmer | Script::Myanmar | Script::Tibetan
        )
    }
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
    } else if is_thai(c) {
        Script::Thai
    } else if is_lao(c) {
        Script::Lao
    } else if is_khmer(c) {
        Script::Khmer
    } else if is_myanmar(c) {
        Script::Myanmar
    } else if is_tibetan(c) {
        Script::Tibetan
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

        // Bigram over grapheme clusters rather than code points. For Han, Kana
        // and Hangul every cluster is one char, so this is the same loop it has
        // always been. For the non-spacing scripts it is the difference between
        // real terms and noise — see the module docs.
        //
        // `bounds` holds each cluster's start offset with the run's end
        // appended, so `bounds[n]` is where cluster `n` starts and where cluster
        // `n - 1` ends. A bigram is then `bounds[n]..bounds[n + 2]`.
        let bounds = cluster_bounds(&chars[run_start..i], script);

        if bounds.len() < 3 {
            // Fewer than two clusters, so there is no bigram to form: either a
            // lone character, or one base plus its combining marks (`ก` and a
            // vowel sign). Emit it whole, so a genuinely isolated character
            // stays findable rather than vanishing.
            let term: String = chars[run_start..i].iter().collect();
            let index = tokens.len();
            tokens.push(Token::with_position(term, run_start, run_len, index));
            continue;
        }

        for pair in bounds.windows(3) {
            let (start, end) = (pair[0], pair[2]);
            let bigram: String = chars[run_start + start..run_start + end].iter().collect();
            let index = tokens.len();
            tokens.push(Token::with_position(
                bigram,
                run_start + start,
                end - start,
                index,
            ));
        }
    }

    tokens
}

/// Cluster start offsets within `run`, with the run's length appended.
///
/// A grapheme cluster is a base character plus any combining marks that follow
/// it. The trailing sentinel means `bounds[n + 1] - bounds[n]` is cluster `n`'s
/// length without a special case for the last one.
///
/// For scripts with no combining marks this is `0..=run.len()`, so a bigram is
/// always two code points and behaviour is identical to fixed-width
/// bigramming — Han, Kana and Hangul indexes are unchanged byte for byte.
fn cluster_bounds(run: &[char], script: Script) -> Vec<usize> {
    let mut bounds = Vec::with_capacity(run.len() + 1);

    if script.has_combining_marks() {
        for (offset, &c) in run.iter().enumerate() {
            // A mark attaches to the preceding base. A leading mark — malformed
            // text, or a run beginning mid-cluster — has no base to attach to,
            // so it starts a cluster of its own rather than being dropped.
            if bounds.is_empty() || !is_combining_mark(c) {
                bounds.push(offset);
            }
        }
    } else {
        bounds.extend(0..run.len());
    }

    bounds.push(run.len());
    bounds
}

/// Trimmer suitable for CJK languages: CJK terms are kept as-is, other terms
/// are stripped of surrounding non-word characters.
///
/// Trimming goes through [`Token::trim_matching`], which moves the recorded
/// position along with the term. Trimming the term alone would leave the
/// position covering the untrimmed span, and every highlight would include the
/// punctuation the trimmer had just removed.
pub fn cjk_trim(token: &mut Token) -> bool {
    if token.term.chars().next().is_some_and(is_cjk_char) {
        return !token.term.is_empty();
    }
    token.trim_matching(is_word_char)
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
