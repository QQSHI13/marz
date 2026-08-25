//! Chinese language implementation.
//!
//! Han runs are tokenized into overlapping character bigrams, Latin runs by
//! separator. See [`crate::languages::cjk`] for why bigrams — and why not
//! unigrams — and for the dictionary-free rationale.

use crate::language::Language;
use crate::languages::cjk::{cjk_trim, tokenize_cjk, Script, CJK_SEPARATORS};
use crate::token::Token;

/// Chinese language configuration.
#[derive(Debug, Clone, Default)]
pub struct Chinese;

impl Language for Chinese {
    fn code(&self) -> &str {
        "zh"
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        tokenize_cjk(text, &[Script::Han])
    }

    fn trim(&self, token: &mut Token) -> bool {
        cjk_trim(token)
    }

    fn is_stop_word(&self, _term: &str) -> bool {
        // Bigrams make a unigram stop-word list meaningless, and a bigram stop
        // list would need corpus statistics. IDF already discounts the common
        // bigrams, which is the right mechanism.
        false
    }

    fn stem(&self, term: &str) -> String {
        // Chinese is not inflected; there is nothing to stem.
        term.to_string()
    }

    fn separator_chars(&self) -> &str {
        CJK_SEPARATORS
    }

    fn is_ngram_script(&self, c: char) -> bool {
        crate::languages::cjk::script_of(c) == Script::Han
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(lang: &Chinese, text: &str) -> Vec<String> {
        lang.tokenize(text).iter().map(|t| t.term.clone()).collect()
    }

    #[test]
    fn chinese_tokenizes_bigrams_only() {
        let t = terms(&Chinese, "中文搜索");
        assert_eq!(t, ["中文", "文搜", "搜索"]);
    }

    #[test]
    fn chinese_keeps_ascii_words() {
        let t = terms(&Chinese, "使用 rust 编程");
        assert!(t.contains(&"rust".to_string()), "got {t:?}");
    }

    #[test]
    fn chinese_folds_fullwidth() {
        let t = terms(&Chinese, "ＲＵＳＴ语言");
        assert!(t.contains(&"rust".to_string()), "got {t:?}");
    }
}
