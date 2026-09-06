//! Languages written without spaces and without script alternation.
//!
//! Thai, Lao, Khmer, Burmese and Tibetan share a problem that neither of Marz's
//! existing code paths solved. They have no word spaces, so the separator
//! tokenizer sees one enormous token: `การค้นหาข้อมูล` — "searching for
//! information" — came out as a single term, and no query shorter than that
//! exact string could match it. And unlike Japanese they do not alternate
//! scripts at morpheme boundaries, so the trick that recovers word boundaries
//! for `検索エンジン` finds nothing to split on here.
//!
//! So they get bigrams, for the same reason Chinese does: it is the
//! dictionary-free option, and phrase verification over positions recovers the
//! precision that bigrams alone would lose.
//!
//! # Why clusters and not code points
//!
//! These scripts stack combining marks — vowel signs, tone marks, viramas — onto
//! a base consonant. `การค้นหาข้อมูล` is 14 code points but 11 clusters, because
//! `ค้`, `ข้` and `มู` are each a consonant plus a mark.
//!
//! Bigramming by code point splits those apart, producing terms like `"้น"` —
//! a mark followed by a consonant. That is not a word fragment or a syllable; it
//! is the tail of one character glued to the head of the next, and it matches
//! nothing a reader would type. Bigramming by cluster gives `ค้น`, `ข้อ`, `มูล`:
//! units that appear in real queries.
//!
//! # No stemming
//!
//! Snowball has no algorithms for these languages, and none of them inflect in
//! the suffix-stripping way a Snowball stemmer handles. `stem` is the identity
//! function, as it is for Chinese and Japanese.

use crate::language::Language;
use crate::languages::cjk::{cjk_trim, script_of, tokenize_cjk, Script, CJK_SEPARATORS};
use crate::token::Token;

/// A language written in a single non-spacing script.
///
/// One struct for all five: they differ only in which script they bigram, and
/// none of them needs stemming or stop words.
pub struct NonSpacing {
    code: &'static str,
    scripts: &'static [Script],
}

impl NonSpacing {
    /// Create a language that bigrams `scripts`.
    pub const fn new(code: &'static str, scripts: &'static [Script]) -> Self {
        Self { code, scripts }
    }
}

impl Language for NonSpacing {
    fn code(&self) -> &str {
        self.code
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        tokenize_cjk(text, self.scripts)
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
        self.scripts.contains(&script_of(c))
    }
}

/// Scripts bigrammed for Thai.
pub const TH_SCRIPTS: &[Script] = &[Script::Thai];
/// Scripts bigrammed for Lao.
pub const LO_SCRIPTS: &[Script] = &[Script::Lao];
/// Scripts bigrammed for Khmer.
pub const KM_SCRIPTS: &[Script] = &[Script::Khmer];
/// Scripts bigrammed for Burmese.
pub const MY_SCRIPTS: &[Script] = &[Script::Myanmar];
/// Scripts bigrammed for Tibetan.
///
/// Tibetan marks syllable boundaries with the tsheg `་`, which the separator
/// path would treat as ordinary text. It is outside the combining-mark ranges,
/// so it becomes its own cluster and bigrams span it — which is correct: a
/// bigram of two syllables is exactly the unit wanted.
pub const BO_SCRIPTS: &[Script] = &[Script::Tibetan];

/// Every non-spacing language, as `(code, scripts)`.
pub const NON_SPACING_LANGUAGES: &[(&str, &[Script])] = &[
    ("th", TH_SCRIPTS),
    ("lo", LO_SCRIPTS),
    ("km", KM_SCRIPTS),
    ("my", MY_SCRIPTS),
    ("bo", BO_SCRIPTS),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::languages::cjk::is_combining_mark;

    fn lang(code: &str) -> NonSpacing {
        let (found, scripts) = NON_SPACING_LANGUAGES
            .iter()
            .find(|(c, _)| *c == code)
            .expect("code in table");
        NonSpacing::new(found, scripts)
    }

    fn terms(code: &str, text: &str) -> Vec<String> {
        lang(code)
            .tokenize(text)
            .into_iter()
            .map(|t| t.term)
            .collect()
    }

    /// The bug this module exists to fix: a phrase used to be one token.
    #[test]
    fn a_thai_phrase_is_not_one_token() {
        let t = terms("th", "การค้นหาข้อมูล");
        assert!(
            t.len() > 5,
            "expected many bigrams, got {} token(s): {t:?}",
            t.len()
        );
    }

    /// Words a person would actually search for must appear as terms.
    ///
    /// `ค้น` (search), `หา` (find) and `ข้อ` (the start of ข้อมูล, information)
    /// are each a cluster pair inside the phrase, so each is an emitted bigram.
    #[test]
    fn real_thai_words_are_findable() {
        let t = terms("th", "การค้นหาข้อมูล");
        for word in ["ค้น", "หา", "ข้อ"] {
            assert!(t.contains(&word.to_string()), "{word} missing from {t:?}");
        }
    }

    /// No term may begin with a combining mark.
    ///
    /// This is the property that distinguishes cluster bigrams from code-point
    /// bigrams, and it is what would break first if a script's mark ranges were
    /// wrong or if Unicode added marks to a block. Every language is checked,
    /// not just Thai.
    #[test]
    fn no_bigram_starts_with_a_combining_mark() {
        let samples = [
            ("th", "การค้นหาข้อมูลเครื่องมือค้นหา"),
            ("lo", "ການຄົ້ນຫາຂໍ້ມູນ"),
            ("km", "ការស្វែងរកព័ត៌មាន"),
            ("my", "အချက်အလက်ရှာဖွေခြင်း"),
            ("bo", "བཙལ་འཚོལ་ཞིབ་འཇུག"),
        ];
        for (code, text) in samples {
            for term in terms(code, text) {
                let first = term.chars().next().expect("non-empty term");
                assert!(
                    !is_combining_mark(first),
                    "{code}: term {term:?} starts with the combining mark {first:?}, \
                     which is not a linguistic unit"
                );
            }
        }
    }

    /// Bigrams must overlap, which is what lets phrase verification reassemble
    /// them. Consecutive tokens' start offsets differ by exactly one cluster.
    #[test]
    fn bigrams_overlap_by_one_cluster() {
        let tokens = lang("th").tokenize("การค้นหาข้อมูล");
        let positions: Vec<(usize, usize)> = tokens.iter().filter_map(|t| t.position()).collect();
        assert!(positions.len() > 5);
        for pair in positions.windows(2) {
            let ((start, len), (next, _)) = (pair[0], pair[1]);
            assert!(
                next > start && next < start + len,
                "tokens at {start} (len {len}) and {next} do not overlap"
            );
        }
    }

    /// A single cluster has no bigram, and must survive rather than vanish.
    #[test]
    fn a_lone_cluster_is_emitted_whole() {
        // `ก` plus a vowel sign: one cluster, two code points.
        let t = terms("th", "กิ");
        assert_eq!(t, ["กิ"]);
    }

    /// Latin text mixed into a non-spacing document still tokenizes on spaces.
    #[test]
    fn embedded_latin_is_word_tokenized() {
        let t = terms("th", "ค้นหา marz search");
        assert!(t.contains(&"marz".to_string()), "got {t:?}");
        assert!(t.contains(&"search".to_string()), "got {t:?}");
    }

    #[test]
    fn every_language_tokenizes_its_own_script() {
        let samples = [
            ("th", "การค้นหา"),
            ("lo", "ການຄົ້ນຫາ"),
            ("km", "ការស្វែងរក"),
            ("my", "ရှာဖွေခြင်း"),
            ("bo", "བཙལ་འཚོལ"),
        ];
        for (code, text) in samples {
            let t = terms(code, text);
            assert!(
                t.len() >= 2,
                "{code}: {text:?} produced {t:?}, which is not segmented"
            );
        }
    }
}
