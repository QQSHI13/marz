//! The single source of truth for which language codes exist.
//!
//! Before this module the code list lived in three places — the Python binding,
//! the WASM binding, and an example — each with its own `match`. At four
//! languages that was merely redundant. At forty-five it guarantees drift: a
//! language added to one binding and not the other produces indexes that one
//! runtime can build and the other cannot load, which surfaces as an empty
//! search result rather than an error.
//!
//! So resolution lives here, and the bindings call [`resolve`].
//!
//! # Exact match or fallback
//!
//! [`resolve`] never fails. An unrecognized code gets
//! [`Generic`](crate::languages::generic::Generic) — whitespace and punctuation
//! tokenization, no stemming — because that is a working search index for the
//! many languages Snowball has no algorithm for, and a hard error would block
//! more of those than it would catch typos.
//!
//! But it reports which happened. [`Resolved::exact`] is false for a fallback,
//! and every binding turns that into a warning naming the code: a `UserWarning`
//! in Python, a `console.warn` in the browser. A typo like `"engish"` is then
//! visible in the build output instead of quietly producing an unstemmed index.

use std::sync::Arc;

use crate::language::LanguageRef;
use crate::languages::generic::Generic;
use crate::languages::nonspacing::{NonSpacing, NON_SPACING_LANGUAGES};
use crate::languages::snowball::{SnowballLanguage, SNOWBALL_LANGUAGES};
use crate::languages::{Chinese, English, Japanese, Korean};

/// The outcome of resolving a language code.
pub struct Resolved {
    /// The language to analyse text with.
    pub language: LanguageRef,
    /// Whether `code` named a language Marz actually implements.
    ///
    /// `false` means the code fell back to generic tokenization. Callers should
    /// say so — see the module docs.
    pub exact: bool,
}

/// Resolve a language code to its analysis rules.
///
/// Codes are matched exactly and case-sensitively, after trimming surrounding
/// whitespace: they come from build configuration and index headers, not from
/// user input, and accepting `"EN"` or `"en-US"` would mean guessing which of
/// several plausible normalizations a caller intended.
///
/// Never fails. Check [`Resolved::exact`] to distinguish a real language from
/// the fallback.
pub fn resolve(code: &str) -> Resolved {
    let code = code.trim();

    if let Some(language) = exact(code) {
        return Resolved {
            language,
            exact: true,
        };
    }

    Resolved {
        // The fallback carries the requested code, so an index built for `vi`
        // reports `vi` in its header rather than a name nobody asked for.
        language: Arc::new(Generic::new(code)),
        exact: false,
    }
}

/// Resolve a possibly multi-language code to its analysis rules.
///
/// A comma-separated list (`"en,ja"`) builds a [`MultiLanguage`] running each
/// member's tokenizer, so one index can serve translated pages. A single code
/// behaves exactly like [`resolve`]. The comma is unambiguous: no language
/// code contains one (unlike `-`, which variant codes like `en-snowball`
/// already use — which is why [`MultiLanguage`] joins with `,`).
///
/// Never fails. `exact` is true only when every member resolved exactly.
pub fn resolve_multi(code: &str) -> Resolved {
    use crate::language::MultiLanguage;

    let mut parts: Vec<&str> = code
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    // Deduplicate preserving order: `"en,en"` is one English tokenizer run
    // twice the work, not two configurations.
    let mut seen = std::collections::HashSet::new();
    parts.retain(|part| seen.insert(*part));
    if parts.len() <= 1 {
        return resolve(parts.first().copied().unwrap_or(""));
    }

    let mut languages = Vec::with_capacity(parts.len());
    let mut exact = true;
    for part in parts {
        let resolved = resolve(part);
        exact &= resolved.exact;
        languages.push(resolved.language);
    }
    Resolved {
        language: Arc::new(MultiLanguage::new(languages)),
        exact,
    }
}

/// Resolve a code, or `None` if Marz has no implementation for it.
fn exact(code: &str) -> Option<LanguageRef> {
    // The hand-written languages come first: `en` must reach the Porter
    // stemmer that `tests/golden.rs` pins against lunr.js, never Snowball's
    // Porter2. See `languages::snowball` for why they differ.
    match code {
        "en" => return Some(Arc::new(English)),
        "zh" => return Some(Arc::new(Chinese)),
        "ja" => return Some(Arc::new(Japanese)),
        "ko" => return Some(Arc::new(Korean)),
        _ => {}
    }

    if let Some((found, stem_fn)) = SNOWBALL_LANGUAGES.iter().find(|(c, _)| *c == code) {
        return Some(Arc::new(SnowballLanguage::new(found, *stem_fn)));
    }

    if let Some((found, scripts)) = NON_SPACING_LANGUAGES.iter().find(|(c, _)| *c == code) {
        return Some(Arc::new(NonSpacing::new(found, scripts)));
    }

    None
}

/// Every language code Marz implements, sorted.
///
/// This is what the bindings' `languages()` reports. Sorting is for the reader:
/// the three source tables are each in their own order, and a caller printing
/// forty-five codes wants them findable.
pub fn codes() -> Vec<&'static str> {
    let mut all: Vec<&'static str> = HAND_WRITTEN
        .iter()
        .copied()
        .chain(SNOWBALL_LANGUAGES.iter().map(|(c, _)| *c))
        .chain(NON_SPACING_LANGUAGES.iter().map(|(c, _)| *c))
        .collect();
    all.sort_unstable();
    all
}

/// Codes backed by a hand-written implementation rather than a table.
///
/// Kept beside [`exact`]'s `match` so that adding one without adding it here
/// fails [`tests::every_code_resolves_exactly`].
const HAND_WRITTEN: &[&str] = &["en", "zh", "ja", "ko"];

/// Split a possibly multi-language code into canonical member codes.
///
/// Trims whitespace, drops empties, preserves order: `"en, ja"` and `"en,ja"`
/// are the same configuration, while `"ja,en"` is not (member order affects
/// stemming). Used to compare a requested code against a stored one without
/// tripping on formatting.
pub fn canonical_parts(code: &str) -> Vec<String> {
    code.split(',')
        .map(|part| part.trim().to_string())
        .filter(|part| !part.is_empty())
        .collect()
}

/// Whether `code` names a language Marz implements.
///
/// Cheaper than [`resolve`] when the answer is all that is wanted, and it does
/// not allocate a language that gets thrown away.
pub fn is_supported(code: &str) -> bool {
    let code = code.trim();
    HAND_WRITTEN.contains(&code)
        || SNOWBALL_LANGUAGES.iter().any(|(c, _)| *c == code)
        || NON_SPACING_LANGUAGES.iter().any(|(c, _)| *c == code)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every advertised code must resolve to a real language.
    ///
    /// A code in `codes()` that falls back is worse than one that is missing:
    /// `languages()` promises it works, and the fallback silently drops
    /// stemming.
    #[test]
    fn every_code_resolves_exactly() {
        for code in codes() {
            let r = resolve(code);
            assert!(r.exact, "advertised code {code:?} fell back to generic");
            assert_eq!(r.language.code(), code, "{code:?} reports a different code");
        }
    }

    /// `codes()` and `is_supported` must agree, in both directions.
    #[test]
    fn is_supported_matches_codes() {
        for code in codes() {
            assert!(is_supported(code), "{code:?} is listed but not supported");
        }
        assert!(!is_supported("engish"));
        assert!(!is_supported(""));
    }

    #[test]
    fn codes_are_unique() {
        let all = codes();
        let mut deduped = all.clone();
        deduped.dedup();
        assert_eq!(
            all.len(),
            deduped.len(),
            "a code appears in more than one table: {all:?}"
        );
    }

    /// The count is not asserted, but the scale is: this module exists because
    /// four hardcoded codes became forty-something.
    ///
    /// Only the default build advertises forty-five. A trimmed build is
    /// *supposed* to report fewer, so counting there would fail for the feature
    /// gates working correctly.
    #[test]
    #[cfg(feature = "all-languages")]
    fn all_three_families_are_present() {
        let all = codes();
        assert!(all.len() > 40, "only {} codes: {all:?}", all.len());
        for code in ["en", "zh", "ja", "ko", "de", "ru", "tr", "th", "km", "bo"] {
            assert!(all.contains(&code), "{code} missing from {all:?}");
        }
    }

    /// The hand-written languages and the non-spacing scripts have no feature to
    /// disable, so they are present in every build — including one that asks for
    /// no stemmers at all.
    #[test]
    fn the_ungated_languages_survive_every_build() {
        let all = codes();
        for code in ["en", "zh", "ja", "ko", "th", "km", "lo", "my", "bo"] {
            assert!(all.contains(&code), "{code} missing from {all:?}");
        }
    }

    /// Turning a language off must remove its *stemmer*, not the language.
    ///
    /// This is the property the size lever depends on, and it fails silently:
    /// `default-features = false` on a dependent is ignored unless
    /// `[workspace.dependencies]` declares it too, so a build asking for one
    /// language can link all thirty-seven and every other test still passes. CI
    /// catches that as a byte count; this catches it with a name.
    #[test]
    #[cfg(not(feature = "de"))]
    fn a_disabled_language_still_resolves_but_stops_stemming() {
        let r = resolve("de");
        assert!(!r.exact, "de was disabled but still resolved exactly");
        // Text is still indexable — just unstemmed.
        assert_eq!(r.language.stem("suchmaschinen"), "suchmaschinen");
        assert_eq!(r.language.tokenize("die suchmaschine").len(), 2);
        assert!(
            !codes().contains(&"de"),
            "a disabled code is still advertised"
        );
    }

    /// `en` must reach the hand-written Porter, not Snowball's Porter2. The two
    /// disagree on `died`, and `tests/golden.rs` pins the lunr.js answer.
    #[test]
    fn en_uses_the_lunr_compatible_stemmer() {
        assert_eq!(resolve("en").language.stem("died"), "di");
    }

    /// The other half of the same guarantee: Porter2 ships, reachable, and is a
    /// genuinely different stemmer — which is why it does not get to be `en`.
    #[test]
    #[cfg(feature = "en-snowball")]
    fn porter2_ships_under_its_own_code() {
        assert_eq!(resolve("en-snowball").language.stem("died"), "die");
    }

    #[test]
    fn unknown_code_falls_back_and_says_so() {
        let r = resolve("engish");
        assert!(!r.exact);
        // The requested code survives, because the binary header stores it.
        assert_eq!(r.language.code(), "engish");
        // Tokenization still works; only stemming is absent.
        assert_eq!(r.language.stem("running"), "running");
        assert_eq!(r.language.tokenize("hello world").len(), 2);
    }

    #[test]
    fn comma_list_builds_a_multi_language() {
        let r = resolve_multi("en,ja");
        assert!(r.exact);
        assert_eq!(r.language.code(), "en,ja");
        // Each side tokenizes by its own rules in one index.
        let terms: Vec<String> = r
            .language
            .tokenize("hello 検索")
            .into_iter()
            .map(|t| t.term)
            .collect();
        assert!(terms.contains(&"hello".to_string()), "got {terms:?}");
        assert!(terms.contains(&"検索".to_string()), "got {terms:?}");
    }

    #[test]
    #[cfg(feature = "de")]
    fn multi_exactness_requires_every_member() {
        assert!(resolve_multi("en,de").exact);
        assert!(!resolve_multi("en,engish").exact);
        // Single codes behave exactly like `resolve`.
        assert!(resolve_multi("en").exact);
        assert!(!resolve_multi("engish").exact);
        assert_eq!(resolve_multi("").language.code(), "");
    }

    /// Vietnamese is the case the fallback is *for*, not a typo.
    #[test]
    fn a_real_unstemmed_language_works() {
        let r = resolve("vi");
        assert!(!r.exact, "vi has no Snowball algorithm");
        let terms: Vec<String> = r
            .language
            .tokenize("công cụ tìm kiếm")
            .into_iter()
            .map(|t| t.term)
            .collect();
        assert_eq!(terms, ["công", "cụ", "tìm", "kiếm"]);
    }

    #[test]
    #[cfg(feature = "de")]
    fn surrounding_whitespace_is_ignored() {
        let r = resolve("  de  ");
        assert!(r.exact, "a padded code must still resolve");
        assert_eq!(r.language.code(), "de");
    }

    /// Case is significant, and the fallback keeps the caller's spelling so the
    /// warning names what they actually passed.
    #[test]
    fn case_is_not_folded() {
        let r = resolve("JA");
        assert!(!r.exact);
        assert_eq!(r.language.code(), "JA");
    }

    /// The empty string is a configuration bug, not a language. It must not
    /// panic, and it must not silently become English.
    #[test]
    fn empty_code_falls_back() {
        let r = resolve("");
        assert!(!r.exact);
        assert_eq!(r.language.code(), "");
    }

    /// Non-spacing codes must reach `NonSpacing`, not the generic fallback —
    /// that is the difference between a segmented Thai index and a one-token
    /// one.
    #[test]
    fn thai_reaches_the_cluster_bigrammer() {
        let r = resolve("th");
        assert!(r.exact);
        assert!(
            r.language.tokenize("การค้นหาข้อมูล").len() > 5,
            "th resolved to something that does not segment Thai"
        );
    }
}
