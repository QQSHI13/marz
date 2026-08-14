//! Japanese language implementation.
//!
//! Uses the same CJK n-gram approach as Chinese, treating Han, Hiragana, and
//! Katakana as script characters. This is a lightweight, fully offline
//! alternative to dictionary-based morphological analyzers.

use crate::language::Language;
use crate::languages::cjk::{cjk_trim, is_cjk_ideograph, is_hiragana, is_katakana, tokenize_cjk};
use crate::token::Token;

/// Japanese language configuration.
#[derive(Debug, Clone, Default)]
pub struct Japanese;

fn is_japanese_char(c: char) -> bool {
    is_cjk_ideograph(c) || is_hiragana(c) || is_katakana(c)
}

impl Language for Japanese {
    fn code(&self) -> &str {
        "ja"
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        tokenize_cjk(text, is_japanese_char)
    }

    fn trim(&self, token: &mut Token) -> bool {
        cjk_trim(token)
    }

    fn is_stop_word(&self, _term: &str) -> bool {
        false
    }

    fn stem(&self, term: &str) -> String {
        term.to_string()
    }

    fn separator_chars(&self) -> &str {
        " \t\n\r\x0C\x0B\x0D\u{00A0}"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn japanese_tokenizes_mixed_script() {
        let ja = Japanese;
        let tokens = ja.tokenize("日本語の検索");
        let terms: Vec<_> = tokens.iter().map(|t| t.term.clone()).collect();
        assert!(terms.contains(&"日本".to_string()));
        assert!(terms.contains(&"検索".to_string()));
    }
}
