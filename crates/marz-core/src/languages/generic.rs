//! Fallback tokenization for languages with no dedicated implementation.
//!
//! Marz knows ~45 language codes, and the world has rather more. Vietnamese,
//! Hebrew, Ukrainian and Welsh all tokenize perfectly well on whitespace and
//! punctuation — they simply have no Snowball algorithm, so there is nothing to
//! stem with. This is what they get, and it is a genuinely useful search index:
//! everything except suffix folding works.
//!
//! It is also what a *typo* gets. `"engish"` resolves here rather than erroring,
//! which is a deliberate trade: with this many codes, a hard error would block
//! more working languages than it would catch typos. But silence would be worse
//! than either, so resolution reports which case happened — see
//! [`crate::languages::registry::resolve`] — and the bindings warn.
//!
//! # Why it carries the requested code
//!
//! `code()` returns the code the caller asked for, not `"generic"`. The binary
//! format stores `language.code()` in its header
//! (`crate::binary::writer`), and that string is what a later `load` resolves
//! against. Returning `"generic"` would erase the distinction between a
//! Vietnamese index and a Hebrew one, and an index built for `vi` would report
//! itself as something no caller ever asked for.

use crate::language::Language;
use crate::token::Token;
use crate::tokenizer::tokenize_with_separator;

/// Separators for a language written with spaces.
///
/// The same set the Snowball languages use, including the hyphen, so
/// `built-in` indexes as `built` and `in`.
const GENERIC_SEPARATORS: &str = " \t\n\r\x0C\x0B\x0D\u{00A0}-";

/// A language that tokenizes on whitespace and punctuation and does not stem.
pub struct Generic {
    code: String,
}

impl Generic {
    /// Create a generic language reporting `code`.
    pub fn new(code: impl Into<String>) -> Self {
        Self { code: code.into() }
    }
}

impl Language for Generic {
    fn code(&self) -> &str {
        &self.code
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        tokenize_with_separator(text, GENERIC_SEPARATORS)
    }

    fn trim(&self, token: &mut Token) -> bool {
        token.trim_matching(|c| c.is_alphanumeric() || c == '_')
    }

    fn is_stop_word(&self, _term: &str) -> bool {
        false
    }

    fn stem(&self, term: &str) -> String {
        term.to_string()
    }

    fn separator_chars(&self) -> &str {
        GENERIC_SEPARATORS
    }

    fn pipeline_labels(&self) -> Vec<&'static str> {
        // No stemmer, so the pipeline is shorter than a stemmed language's. The
        // labels are serialized into JSON indexes, so they must describe what
        // actually ran.
        vec!["trimmer", "stopWordFilter"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(code: &str, text: &str) -> Vec<String> {
        Generic::new(code)
            .tokenize(text)
            .into_iter()
            .map(|t| t.term)
            .collect()
    }

    /// The code round-trips, because the binary header stores it.
    #[test]
    fn code_is_the_requested_one() {
        assert_eq!(Generic::new("vi").code(), "vi");
        assert_eq!(Generic::new("engish").code(), "engish");
    }

    /// Vietnamese is the motivating case: no Snowball algorithm, but spaces and
    /// diacritics that must survive tokenization.
    #[test]
    fn vietnamese_tokenizes_on_spaces() {
        let t = terms("vi", "công cụ tìm kiếm");
        assert_eq!(t, ["công", "cụ", "tìm", "kiếm"]);
    }

    /// Hebrew is right-to-left, which is a rendering concern and not a
    /// tokenization one — the separator split works unchanged.
    #[test]
    fn hebrew_tokenizes_on_spaces() {
        let t = terms("he", "מנוע חיפוש");
        assert_eq!(t, ["מנוע", "חיפוש"]);
    }

    #[test]
    fn terms_are_not_stemmed() {
        let g = Generic::new("vi");
        for word in ["running", "installations", "запросы"] {
            assert_eq!(g.stem(word), word, "generic must not stem");
        }
    }

    /// The trimmer strips surrounding punctuation, keeping the diacritics that
    /// `is_alphanumeric` correctly counts as word characters.
    #[test]
    fn punctuation_is_trimmed() {
        let g = Generic::new("vi");
        for (input, expected) in [("(tìm)", "tìm"), ("kiếm.", "kiếm"), ("«מנוע»", "מנוע")] {
            let mut token = Token::new(input);
            assert!(g.trim(&mut token), "{input} was dropped");
            assert_eq!(token.term, expected);
        }
    }

    #[test]
    fn hyphenated_compounds_split() {
        assert_eq!(terms("vi", "built-in"), ["built", "in"]);
    }
}
