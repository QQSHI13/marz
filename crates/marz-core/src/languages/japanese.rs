//! Japanese language implementation.
//!
//! Han, Hiragana and Katakana runs are each bigrammed independently, so no
//! bigram spans a script change. Japanese switches script at morpheme
//! boundaries — `検索エンジン` is Han + Katakana, `日本語の検索` puts the
//! particle `の` in its own Hiragana run — so script segmentation recovers real
//! word boundaries with no dictionary at all. That is why `検索` comes out as a
//! clean term here while a naive whole-string bigrammer would also emit the
//! meaningless `索エ`.

use crate::language::Language;
use crate::languages::cjk::{cjk_trim, script_of, tokenize_cjk, Script, CJK_SEPARATORS};
use crate::token::Token;

/// Japanese language configuration.
#[derive(Debug, Clone, Default)]
pub struct Japanese;

/// Scripts that are bigrammed for Japanese.
const JA_SCRIPTS: &[Script] = &[Script::Han, Script::Hiragana, Script::Katakana];

impl Language for Japanese {
    fn code(&self) -> &str {
        "ja"
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        tokenize_cjk(text, JA_SCRIPTS)
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
        JA_SCRIPTS.contains(&script_of(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(text: &str) -> Vec<String> {
        Japanese
            .tokenize(text)
            .iter()
            .map(|t| t.term.clone())
            .collect()
    }

    #[test]
    fn japanese_tokenizes_mixed_script() {
        let t = terms("日本語の検索");
        // 日本語 (Han) / の (Hiragana) / 検索 (Han)
        assert!(t.contains(&"日本".to_string()), "got {t:?}");
        assert!(t.contains(&"検索".to_string()), "got {t:?}");
        // The lone particle survives as a unigram since its run has length 1.
        assert!(t.contains(&"の".to_string()), "got {t:?}");
    }

    #[test]
    fn no_bigram_spans_han_katakana_boundary() {
        let t = terms("検索エンジン");
        assert_eq!(t, ["検索", "エン", "ンジ", "ジン"]);
        assert!(!t.contains(&"索エ".to_string()), "got {t:?}");
    }

    #[test]
    fn halfwidth_katakana_matches_fullwidth() {
        assert_eq!(terms("ｶﾞｲﾄﾞ"), terms("ガイド"));
    }
}
