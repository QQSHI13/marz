//! Korean language implementation.
//!
//! Hangul syllable blocks are treated as script characters and tokenized into
//! unigrams and overlapping bigrams. Hanja (CJK ideographs) are also handled
//! as script characters for mixed Korean/Chinese documents.

use crate::language::Language;
use crate::languages::cjk::{cjk_trim, is_cjk_ideograph, is_hangul, tokenize_cjk};
use crate::token::Token;

/// Korean language configuration.
#[derive(Debug, Clone, Default)]
pub struct Korean;

fn is_korean_char(c: char) -> bool {
    is_hangul(c) || is_cjk_ideograph(c)
}

impl Language for Korean {
    fn code(&self) -> &str {
        "ko"
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        tokenize_cjk(text, is_korean_char)
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
    fn korean_tokenizes_hangul() {
        let ko = Korean;
        let tokens = ko.tokenize("한국어 검색");
        let terms: Vec<_> = tokens.iter().map(|t| t.term.clone()).collect();
        assert!(terms.contains(&"한국".to_string()));
        assert!(terms.contains(&"국어".to_string()));
    }
}
