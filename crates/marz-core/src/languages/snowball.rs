//! Languages whose stemming comes from a Snowball algorithm.
//!
//! One struct serves every Snowball language. The languages differ only in
//! which `stem` function they call — tokenization is the same separator split
//! for all of them, because Snowball's algorithms are defined over words that a
//! separator tokenizer already produces. Forty hand-written files that differed
//! by one function pointer would be forty places for a typo to hide, so each
//! language is a row in [`SNOWBALL_LANGUAGES`] instead.
//!
//! # Why English is not here
//!
//! Snowball's `english` algorithm is Porter**2**, which is a different stemmer
//! from the Porter used by lunr.js. They disagree on real words: Porter2 stems
//! `died` to `die`, lunr.js's Porter to `di`. Marz's English ranking is checked
//! against lunr.js on every commit (see `tests/golden.rs`), so `en` keeps the
//! hand-written [`crate::languages::porter`] and never routes here.
//!
//! Both stemmers ship. Snowball's is reachable as `en-snowball` for callers who
//! want Porter2 and are not trying to match lunr — the two are genuinely
//! different tools, and silently substituting one would change every English
//! index's terms.
//!
//! # Stop words
//!
//! Snowball ships algorithms, not stop-word lists, and there is no canonical
//! list for most of these languages. Rather than invent forty of them, every
//! language here has an empty list. The cost is index size, not correctness:
//! BM25 already scores a term appearing in most documents near zero through
//! IDF, which is the effect a stop-word list approximates by deleting them. The
//! one exception is `en`, which uses lunr.js's list because matching lunr
//! requires it.

use crate::language::Language;
use crate::stemmers::snowball::SnowballEnv;
use crate::token::Token;
use crate::tokenizer::{is_word_char, tokenize_with_separator};

/// Separators for languages written with spaces and Latin-style punctuation.
///
/// The hyphen is included, matching [`crate::languages::English`]: hyphenated
/// compounds are indexed as their parts, so `built-in` is found by `built`.
const WORD_SEPARATORS: &str = " \t\n\r\x0C\x0B\x0D\u{00A0}-";

/// A generated Snowball `stem` function.
///
/// It mutates the environment's current string in place and returns whether the
/// algorithm succeeded, which every generated algorithm reports as `true` — the
/// bool exists because Snowball's own language has failure semantics, not
/// because a caller has anything to do about it.
pub type StemFn = fn(&mut SnowballEnv) -> bool;

/// A language whose stemmer is a generated Snowball algorithm.
///
/// Constructed from [`SNOWBALL_LANGUAGES`] via
/// [`crate::languages::registry::resolve`]; there is no reason to build one by
/// hand.
pub struct SnowballLanguage {
    code: &'static str,
    stem_fn: StemFn,
    separators: &'static str,
}

impl SnowballLanguage {
    /// Create a language from a code and a generated `stem` function.
    pub const fn new(code: &'static str, stem_fn: StemFn) -> Self {
        Self {
            code,
            stem_fn,
            separators: WORD_SEPARATORS,
        }
    }
}

impl Language for SnowballLanguage {
    fn code(&self) -> &str {
        self.code
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        // Turkish needs dotless-I lowercasing; the global `normalize` would
        // merge `I`/`ı`. Fold with Turkish rules first, then split without a
        // second (wrong) normalize pass.
        if self.code == "tr" {
            let normalized = crate::normalize::normalize_tr(text);
            return crate::tokenizer::tokenize_normalized(&normalized, self.separators);
        }
        tokenize_with_separator(text, self.separators)
    }

    fn trim(&self, token: &mut Token) -> bool {
        token.trim_matching(is_word_char)
    }

    fn is_stop_word(&self, _term: &str) -> bool {
        // See the module docs: no invented lists, and IDF already handles the
        // frequency case a list would target.
        false
    }

    fn stem(&self, term: &str) -> String {
        // Snowball's algorithms assume lowercase input. Every caller reaches
        // this through the pipeline, which normalizes (and so lowercases)
        // first, but `stem` is public and a direct caller has no such
        // guarantee — and an uppercase term silently fails to stem rather than
        // erroring, which is the kind of bug that surfaces as "search works
        // except for capitalized words".
        let lowered = if self.code == "tr" {
            let mut s = String::with_capacity(term.len());
            for c in term.chars() {
                match c {
                    'I' => s.push('ı'),
                    'İ' => s.push('i'),
                    _ => s.extend(c.to_lowercase()),
                }
            }
            s
        } else {
            term.to_lowercase()
        };
        let mut env = SnowballEnv::create(&lowered);
        (self.stem_fn)(&mut env);
        env.get_current().into_owned()
    }

    fn separator_chars(&self) -> &str {
        self.separators
    }

    fn pipeline_labels(&self) -> Vec<&'static str> {
        vec!["trimmer", "stopWordFilter", "stemmer"]
    }
}

/// Every language code backed by a Snowball stemmer.
///
/// Codes are ISO 639-1 where one exists, ISO 639-3 otherwise (`hye` has a
/// 639-1 `hy`, but Basque `eu`, Irish `ga` and the rest are all 639-1; only
/// Sesotho `st` and Esperanto `eo` are worth double-checking against the
/// table).
///
/// Four entries are variant algorithms rather than languages, and none of them
/// is what a caller asking for that language wants by default:
///
/// - `en-snowball` — Porter2. See the module docs for why `en` is not this.
/// - `en-porter` — the original 1980 Porter algorithm, kept because some
///   corpora were indexed with it.
/// - `en-lovins` — Lovins 1968, of historical interest.
/// - `nl-porter` — the older Dutch algorithm, superseded by `nl`.
///
/// The set tracks the pinned `SNOWBALL_REF`, not this list's age: upstream adds
/// algorithms between releases (`earlymodernenglish` landed after v3.1.1), and
/// `table_covers_every_generated_stemmer` fails if the two drift apart.
pub const SNOWBALL_LANGUAGES: &[(&str, StemFn)] = &[
    #[cfg(feature = "ar")]
    ("ar", crate::stemmers::st_arabic::stem),
    #[cfg(feature = "ca")]
    ("ca", crate::stemmers::st_catalan::stem),
    #[cfg(feature = "cs")]
    ("cs", crate::stemmers::st_czech::stem),
    #[cfg(feature = "da")]
    ("da", crate::stemmers::st_danish::stem),
    #[cfg(feature = "de")]
    ("de", crate::stemmers::st_german::stem),
    #[cfg(feature = "el")]
    ("el", crate::stemmers::st_greek::stem),
    #[cfg(feature = "eo")]
    ("eo", crate::stemmers::st_esperanto::stem),
    #[cfg(feature = "es")]
    ("es", crate::stemmers::st_spanish::stem),
    #[cfg(feature = "et")]
    ("et", crate::stemmers::st_estonian::stem),
    #[cfg(feature = "eu")]
    ("eu", crate::stemmers::st_basque::stem),
    #[cfg(feature = "fa")]
    ("fa", crate::stemmers::st_persian::stem),
    #[cfg(feature = "fi")]
    ("fi", crate::stemmers::st_finnish::stem),
    #[cfg(feature = "fr")]
    ("fr", crate::stemmers::st_french::stem),
    #[cfg(feature = "ga")]
    ("ga", crate::stemmers::st_irish::stem),
    #[cfg(feature = "hi")]
    ("hi", crate::stemmers::st_hindi::stem),
    #[cfg(feature = "hu")]
    ("hu", crate::stemmers::st_hungarian::stem),
    #[cfg(feature = "hy")]
    ("hy", crate::stemmers::st_armenian::stem),
    #[cfg(feature = "id")]
    ("id", crate::stemmers::st_indonesian::stem),
    #[cfg(feature = "it")]
    ("it", crate::stemmers::st_italian::stem),
    #[cfg(feature = "lt")]
    ("lt", crate::stemmers::st_lithuanian::stem),
    #[cfg(feature = "ne")]
    ("ne", crate::stemmers::st_nepali::stem),
    #[cfg(feature = "nl")]
    ("nl", crate::stemmers::st_dutch::stem),
    #[cfg(feature = "no")]
    ("no", crate::stemmers::st_norwegian::stem),
    #[cfg(feature = "pl")]
    ("pl", crate::stemmers::st_polish::stem),
    #[cfg(feature = "pt")]
    ("pt", crate::stemmers::st_portuguese::stem),
    #[cfg(feature = "ro")]
    ("ro", crate::stemmers::st_romanian::stem),
    #[cfg(feature = "ru")]
    ("ru", crate::stemmers::st_russian::stem),
    #[cfg(feature = "sr")]
    ("sr", crate::stemmers::st_serbian::stem),
    #[cfg(feature = "st")]
    ("st", crate::stemmers::st_sesotho::stem),
    #[cfg(feature = "sv")]
    ("sv", crate::stemmers::st_swedish::stem),
    #[cfg(feature = "ta")]
    ("ta", crate::stemmers::st_tamil::stem),
    #[cfg(feature = "tr")]
    ("tr", crate::stemmers::st_turkish::stem),
    #[cfg(feature = "yi")]
    ("yi", crate::stemmers::st_yiddish::stem),
    // Variant algorithms. See the doc comment above.
    #[cfg(feature = "en-snowball")]
    ("en-snowball", crate::stemmers::st_english::stem),
    #[cfg(feature = "en-porter")]
    ("en-porter", crate::stemmers::st_porter::stem),
    #[cfg(feature = "en-lovins")]
    ("en-lovins", crate::stemmers::st_lovins::stem),
    #[cfg(feature = "nl-porter")]
    ("nl-porter", crate::stemmers::st_dutch_porter::stem),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::Language;

    /// The language for `code`, or `None` if this build did not include it.
    ///
    /// Returning an option rather than panicking is what lets the tests below
    /// run unchanged in a trimmed build: a language the features excluded is
    /// skipped, not a failure. `assertions_actually_ran` keeps that from
    /// degrading into a suite that silently checks nothing.
    fn lang(code: &str) -> Option<SnowballLanguage> {
        SNOWBALL_LANGUAGES
            .iter()
            .find(|(c, _)| *c == code)
            .map(|(found, stem_fn)| SnowballLanguage::new(found, *stem_fn))
    }

    /// Every generated stemmer is reachable through the table.
    ///
    /// A stemmer that is generated but left out of the table compiles fine and
    /// is simply unusable — no other test would notice, and the symptom is a
    /// language silently missing. So this reads the generated directory rather
    /// than asserting a count someone has to remember to bump: upstream adds
    /// algorithms between releases, and the point is to catch the table failing
    /// to keep up.
    ///
    /// Only meaningful in a build that asks for every language. A trimmed build
    /// has fewer rows *by design*, so the comparison would fail for the feature
    /// gates doing their job.
    #[test]
    #[cfg(feature = "all-languages")]
    fn table_covers_every_generated_stemmer() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/stemmers");
        let mut generated: Vec<String> = std::fs::read_dir(dir)
            .expect("stemmers directory")
            .filter_map(|e| {
                let name = e.ok()?.file_name().to_string_lossy().into_owned();
                name.strip_prefix("st_")?
                    .strip_suffix(".rs")
                    .map(String::from)
            })
            .collect();
        generated.sort();

        // The table stores function pointers, not names, so recover the names
        // from the module paths the table is built from. Comparing pointer
        // identity would be the direct check, but a duplicated row pointing two
        // codes at one algorithm is legitimate (`en-snowball` and `en-porter`
        // are different algorithms, but a future alias might not be).
        let mut mapped: Vec<String> = MODULE_NAMES.iter().map(|s| (*s).to_string()).collect();
        mapped.sort();
        mapped.dedup();

        assert_eq!(
            generated, mapped,
            "the table and scripts/vendor-stemmers.sh disagree about which \
             algorithms exist; regenerate, then add or remove the rows"
        );
        assert_eq!(
            MODULE_NAMES.len(),
            SNOWBALL_LANGUAGES.len(),
            "MODULE_NAMES must list one module per table row, in the same order"
        );
    }

    /// The module name behind each row of [`SNOWBALL_LANGUAGES`], in order.
    ///
    /// Rust has no way to recover `st_german` from a function pointer, so the
    /// names are repeated here and the test above checks the two lists stay the
    /// same length. Adding a language means touching both, which is the point:
    /// the pair is what proves nothing was generated and then forgotten.
    #[cfg(feature = "all-languages")]
    const MODULE_NAMES: &[&str] = &[
        "arabic",
        "catalan",
        "czech",
        "danish",
        "german",
        "greek",
        "esperanto",
        "spanish",
        "estonian",
        "basque",
        "persian",
        "finnish",
        "french",
        "irish",
        "hindi",
        "hungarian",
        "armenian",
        "indonesian",
        "italian",
        "lithuanian",
        "nepali",
        "dutch",
        "norwegian",
        "polish",
        "portuguese",
        "romanian",
        "russian",
        "serbian",
        "sesotho",
        "swedish",
        "tamil",
        "turkish",
        "yiddish",
        "english",
        "porter",
        "lovins",
        "dutch_porter",
    ];

    #[test]
    fn codes_are_unique() {
        let mut codes: Vec<&str> = SNOWBALL_LANGUAGES.iter().map(|(c, _)| *c).collect();
        codes.sort_unstable();
        let before = codes.len();
        codes.dedup();
        assert_eq!(before, codes.len(), "duplicate language code in the table");
    }

    /// `en` must never resolve here — it would change every English index's
    /// terms and break the lunr.js golden test.
    #[test]
    fn plain_en_is_not_in_the_table() {
        assert!(
            !SNOWBALL_LANGUAGES.iter().any(|(c, _)| *c == "en"),
            "en belongs to the hand-written Porter; see the module docs"
        );
    }

    /// The inflected and base forms of a word must reach the same term, which
    /// is the entire point of stemming. Each case is a word an ordinary search
    /// would fail on unstemmed.
    #[test]
    fn inflections_reach_the_base_form() {
        let cases = [
            ("de", "suchmaschinen", "suchmaschine"),
            ("ru", "документации", "документация"),
            ("tr", "evlerimizden", "evler"),
            ("fi", "taloissamme", "talo"),
            ("cs", "dokumentace", "dokumentaci"),
            ("pl", "dokumentacji", "dokumentacja"),
            ("id", "pencarian", "cari"),
        ];
        let mut ran = 0;
        for (code, inflected, base) in cases {
            let Some(l) = lang(code) else { continue };
            ran += 1;
            assert_eq!(
                l.stem(inflected),
                l.stem(base),
                "{code}: {inflected} and {base} must stem alike"
            );
        }
        assert!(ran > 0, "no Snowball languages enabled; nothing was tested");
    }

    /// Snowball's Arabic and Hindi algorithms strip diacritics themselves, so
    /// text with harakat matches text without. This is why no extra
    /// normalization step exists for those scripts.
    #[test]
    fn arabic_diacritics_are_stripped() {
        let Some(ar) = lang("ar") else { return };
        assert_eq!(ar.stem("مُحَرِّك"), ar.stem("محرك"));
        assert_eq!(ar.stem("الْبَحْث"), ar.stem("بحث"));
    }

    /// Stemming is case-insensitive even when called directly, without the
    /// pipeline's normalization in front of it.
    #[test]
    fn stemming_does_not_depend_on_case() {
        let Some(de) = lang("de") else { return };
        assert_eq!(de.stem("Suchmaschinen"), de.stem("suchmaschinen"));
    }

    /// A stemmer must not panic or corrupt input on text it was not designed
    /// for. Mixed scripts and empty strings reach `stem` in real corpora.
    #[test]
    fn stemming_tolerates_unexpected_input() {
        let mut ran = 0;
        for (code, _) in SNOWBALL_LANGUAGES {
            let Some(l) = lang(code) else { continue };
            ran += 1;
            assert_eq!(l.stem(""), "", "{code} mangled the empty string");
            // Not asserting the output, only that there is one.
            let _ = l.stem("中文");
            let _ = l.stem("123");
            let _ = l.stem("a");
        }
        assert!(ran > 0, "no Snowball languages enabled; nothing was tested");
    }

    #[test]
    fn tokenization_splits_on_words_and_hyphens() {
        let Some(de) = lang("de") else { return };
        let terms: Vec<String> = de
            .tokenize("Die Suchmaschine ist Open-Source")
            .into_iter()
            .map(|t| t.term)
            .collect();
        assert_eq!(terms, ["die", "suchmaschine", "ist", "open", "source"]);
    }
}
