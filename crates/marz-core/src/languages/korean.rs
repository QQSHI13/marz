//! Korean language implementation.
//!
//! Korean is written with spaces between eojeol (word + attached particles), so
//! whitespace is a hard token boundary — no bigram ever spans it. Within an
//! eojeol, Hangul is bigrammed, which is what makes the agglutinative
//! morphology searchable without a dictionary:
//!
//! ```text
//! "검색엔진을"  ->  검색 색엔 엔진 진을
//! ```
//!
//! A query for `검색엔진` decomposes to `검색 색엔 엔진`, all present and
//! adjacent, so the document matches despite the attached object particle `을`.
//! Indexing the eojeol as one term would miss it entirely, and indexing
//! unigrams would match almost everything.
//!
//! Hangul and Hanja are separate script runs, so a mixed `한자漢字` sequence
//! never produces the cross-script bigram `자漢`.

use crate::language::Language;
use crate::languages::cjk::{cjk_trim, script_of, tokenize_cjk, Script, CJK_SEPARATORS};
use crate::token::Token;

/// Korean language configuration.
#[derive(Debug, Clone, Default)]
pub struct Korean;

/// Scripts that are bigrammed for Korean: Hangul, plus Hanja for mixed text.
const KO_SCRIPTS: &[Script] = &[Script::Hangul, Script::Han];

impl Language for Korean {
    fn code(&self) -> &str {
        "ko"
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        tokenize_cjk(text, KO_SCRIPTS)
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
        CJK_SEPARATORS
    }

    fn is_ngram_script(&self, c: char) -> bool {
        KO_SCRIPTS.contains(&script_of(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(text: &str) -> Vec<String> {
        Korean
            .tokenize(text)
            .iter()
            .map(|t| t.term.clone())
            .collect()
    }

    #[test]
    fn korean_tokenizes_hangul_bigrams() {
        let t = terms("한국어 검색");
        assert_eq!(t, ["한국", "국어", "검색"]);
    }

    #[test]
    fn no_bigram_spans_whitespace() {
        let t = terms("한국어 검색");
        assert!(!t.contains(&"어검".to_string()), "got {t:?}");
    }

    #[test]
    fn particles_do_not_block_matching() {
        // The query bigrams for 검색엔진 must all appear in the indexed eojeol.
        let indexed = terms("검색엔진을");
        for q in terms("검색엔진") {
            assert!(indexed.contains(&q), "{q} missing from {indexed:?}");
        }
    }

    #[test]
    fn no_bigram_spans_hangul_hanja_boundary() {
        let t = terms("한자漢字");
        assert!(!t.contains(&"자漢".to_string()), "got {t:?}");
    }
}
