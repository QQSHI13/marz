//! Chinese language implementation.
//!
//! Tokenization is script-aware: Han characters are split into unigrams and
//! overlapping bigrams, while Latin text is tokenized by separators. This
//! avoids a heavyweight dictionary while still giving good recall and
//! reasonable phrase adjacency signals.

use crate::language::Language;
use crate::languages::cjk::{cjk_trim, is_cjk_ideograph, tokenize_cjk};
use crate::token::Token;

/// Chinese language configuration.
#[derive(Debug, Clone, Default)]
pub struct Chinese;

impl Language for Chinese {
    fn code(&self) -> &str {
        "zh"
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        tokenize_cjk(text, is_cjk_ideograph)
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
    fn chinese_tokenizes_bigrams_and_unigrams() {
        let zh = Chinese;
        let tokens = zh.tokenize("中文搜索");
        let terms: Vec<_> = tokens.iter().map(|t| t.term.clone()).collect();
        assert!(terms.contains(&"中".to_string()));
        assert!(terms.contains(&"中文".to_string()));
        assert!(terms.contains(&"文搜".to_string()));
        assert!(terms.contains(&"搜索".to_string()));
    }

    #[test]
    fn chinese_keeps_ascii_words() {
        let zh = Chinese;
        let tokens = zh.tokenize("使用 rust 编程");
        let terms: Vec<_> = tokens.iter().map(|t| t.term.clone()).collect();
        assert!(terms.contains(&"rust".to_string()));
    }
}
