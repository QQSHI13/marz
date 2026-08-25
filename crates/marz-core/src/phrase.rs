//! Phrase verification for n-gram-tokenized scripts.
//!
//! # The problem bigrams create
//!
//! Chinese, Japanese and Korean text is indexed as overlapping character
//! bigrams, because segmenting it into words would need a dictionary (see
//! [`crate::languages::cjk`]). That buys recall without a dependency, but it
//! costs precision: a query decomposes into several bigrams, and a document
//! that happens to contain all of them *anywhere* looks like a match.
//!
//! Searching a Japanese Wikipedia corpus for `検索エンジン` ("search engine")
//! returns 72 documents. Only a handful actually discuss search engines; the
//! rest merely contain `検索`, `エン`, `ンジ` and `ジン` in unrelated places —
//! `エンジン` also means a mechanical engine, and `ジン` is gin.
//!
//! # Adjacency as a substitute for a segmenter
//!
//! The bigrams of a contiguous string are themselves contiguous. `検索エンジン`
//! yields tokens at consecutive offsets:
//!
//! ```text
//! 検索  エン  ンジ  ジン
//!  0     2     3     4      <- offsets within the query
//! ```
//!
//! So a document contains the phrase exactly when it contains those same
//! bigrams at offsets with the same *relative* spacing. Verifying that turns a
//! bag-of-bigrams match back into a phrase match, using only the positions
//! already stored for highlighting — no dictionary, no extra index data.
//!
//! # Boost, not filter
//!
//! A verified phrase multiplies the document's score rather than gating it.
//! Filtering would be more precise but changes the query's meaning: default
//! search is disjunctive, and a user typing `機械学習` still wants documents
//! about `学習` ranked below exact matches rather than absent. Boosting fixes
//! the ordering, which is what a user actually observes, while keeping the
//! recall that made bigrams worth choosing.

use std::collections::HashMap;

use crate::language::LanguageRef;
use crate::token::Token;

/// How much a verified phrase match multiplies a clause's contribution.
///
/// Chosen so that an exact phrase match outranks a document that merely
/// contains the same bigrams scattered, without letting a single phrase hit
/// swamp every other signal. Applied per phrase, per field.
pub const PHRASE_BOOST: f64 = 2.0;

/// A run of n-gram terms that were contiguous in the query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phrase {
    /// The terms, in query order.
    pub terms: Vec<String>,
    /// Each term's offset relative to the first term of the phrase.
    ///
    /// For a bigram run these are `0, 1, 2, ...`, but storing them explicitly
    /// keeps verification correct if a script boundary or a dropped stop word
    /// leaves a gap.
    pub offsets: Vec<usize>,
}

impl Phrase {
    /// Whether `term` is part of this phrase.
    pub fn contains(&self, term: &str) -> bool {
        self.terms.iter().any(|t| t == term)
    }
}

/// Extract the phrases from a query clause's tokens.
///
/// A phrase is a maximal run of two or more adjacent tokens whose terms are
/// n-grams of the same script. Tokens are adjacent when their start offsets
/// differ by less than the token's own length — that is, when the n-grams
/// overlap, which is exactly what the CJK tokenizer emits for a contiguous
/// run of characters.
///
/// Returns an empty vector for languages that do not n-gram (the trait method
/// defaults to `false`), so word-tokenized languages pay nothing for this.
pub fn extract_phrases(tokens: &[Token], language: &LanguageRef) -> Vec<Phrase> {
    let mut phrases = Vec::new();
    let mut run: Vec<(String, usize)> = Vec::new();

    let flush = |run: &mut Vec<(String, usize)>, phrases: &mut Vec<Phrase>| {
        if run.len() >= 2 {
            let base = run[0].1;
            phrases.push(Phrase {
                terms: run.iter().map(|(t, _)| t.clone()).collect(),
                offsets: run.iter().map(|(_, p)| p - base).collect(),
            });
        }
        run.clear();
    };

    for token in tokens {
        let is_ngram = token
            .term
            .chars()
            .next()
            .is_some_and(|c| language.is_ngram_script(c));
        let Some((start, len)) = token.position() else {
            flush(&mut run, &mut phrases);
            continue;
        };

        if !is_ngram {
            flush(&mut run, &mut phrases);
            continue;
        }

        // Adjacent means the previous n-gram overlaps this one. A gap larger
        // than the token length means an intervening non-n-gram token or a
        // script change, and the two runs are separate phrases.
        if let Some(&(_, prev_start)) = run.last() {
            if start <= prev_start || start - prev_start >= len.max(1) {
                flush(&mut run, &mut phrases);
            }
        }
        run.push((token.term.clone(), start));
    }
    flush(&mut run, &mut phrases);

    phrases
}

/// Whether `phrase` occurs in a document field, given each term's positions.
///
/// `positions_for` resolves a phrase term to its `(start, length)` positions in
/// the field being checked, or `None` if the term is absent.
///
/// A phrase occurs when some start offset `p` in the field has every term
/// `i` present at `p + offsets[i]`. Absence of any single term is decisive, so
/// the common case — a document that contains only part of the phrase — costs
/// one lookup.
pub fn verify<'a, F>(phrase: &Phrase, mut positions_for: F) -> bool
where
    F: FnMut(&str) -> Option<&'a [(usize, usize)]>,
{
    // Collect every term's start offsets up front. Any missing term rules the
    // phrase out immediately.
    let mut starts: Vec<Vec<usize>> = Vec::with_capacity(phrase.terms.len());
    for term in &phrase.terms {
        let Some(positions) = positions_for(term) else {
            return false;
        };
        if positions.is_empty() {
            return false;
        }
        // Sorted so the alignment check below can binary search. Positions are
        // already appended in ascending order during indexing, but sorting
        // makes the requirement local rather than an assumption about a caller
        // in another module.
        let mut term_starts: Vec<usize> = positions.iter().map(|(start, _)| *start).collect();
        term_starts.sort_unstable();
        term_starts.dedup();
        starts.push(term_starts);
    }

    // Anchor on the rarest term rather than the first, so the candidate loop
    // runs as few times as possible. A phrase's leading bigram is often its
    // most common one.
    let (anchor, _) = starts
        .iter()
        .enumerate()
        .min_by_key(|(_, s)| s.len())
        .expect("phrases have at least two terms");

    let anchor_offset = phrase.offsets[anchor];
    for &anchor_start in &starts[anchor] {
        // Where the phrase would begin if this occurrence is the real anchor.
        let Some(phrase_start) = anchor_start.checked_sub(anchor_offset) else {
            continue;
        };
        let all_present = phrase.offsets.iter().enumerate().all(|(i, offset)| {
            i == anchor || starts[i].binary_search(&(phrase_start + offset)).is_ok()
        });
        if all_present {
            return true;
        }
    }
    false
}

/// Cache of phrase-verification results, keyed by phrase and document field.
///
/// Verification is per (phrase, field, document), but a clause adds a score for
/// every phrase term separately, so without a cache the same check would run
/// once per term.
#[derive(Debug, Default)]
pub struct VerificationCache {
    results: HashMap<(usize, String, String), bool>,
}

impl VerificationCache {
    /// Look up a cached verdict, or compute and store one.
    pub fn verify<'a, F>(
        &mut self,
        phrase_index: usize,
        phrase: &Phrase,
        field: &str,
        doc_ref: &str,
        positions_for: F,
    ) -> bool
    where
        F: FnMut(&str) -> Option<&'a [(usize, usize)]>,
    {
        let key = (phrase_index, field.to_string(), doc_ref.to_string());
        if let Some(&cached) = self.results.get(&key) {
            return cached;
        }
        let verdict = verify(phrase, positions_for);
        self.results.insert(key, verdict);
        verdict
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::languages::{English, Japanese};
    use std::sync::Arc;

    fn phrase(terms: &[&str]) -> Phrase {
        Phrase {
            terms: terms.iter().map(|t| t.to_string()).collect(),
            offsets: (0..terms.len()).collect(),
        }
    }

    /// Build a positions lookup from `(term, starts)` pairs.
    fn positions(pairs: &[(&str, &[usize])]) -> HashMap<String, Vec<(usize, usize)>> {
        pairs
            .iter()
            .map(|(term, starts)| {
                (
                    term.to_string(),
                    starts.iter().map(|s| (*s, 2)).collect::<Vec<_>>(),
                )
            })
            .collect()
    }

    #[test]
    fn japanese_query_yields_one_phrase() {
        let language: LanguageRef = Arc::new(Japanese);
        let tokens = language.tokenize("検索エンジン");
        let phrases = extract_phrases(&tokens, &language);

        // 検索 is Han, エンジン is Katakana: the script change splits the run,
        // so 検索 stands alone and only the katakana bigrams form a phrase.
        assert_eq!(phrases.len(), 1, "got {phrases:?}");
        assert_eq!(phrases[0].terms, ["エン", "ンジ", "ジン"]);
        assert_eq!(phrases[0].offsets, [0, 1, 2]);
    }

    #[test]
    fn longer_han_run_is_one_phrase() {
        let language: LanguageRef = Arc::new(Japanese);
        let tokens = language.tokenize("機械学習");
        let phrases = extract_phrases(&tokens, &language);
        assert_eq!(phrases.len(), 1);
        assert_eq!(phrases[0].terms, ["機械", "械学", "学習"]);
    }

    #[test]
    fn english_query_yields_no_phrases() {
        // Word-tokenized languages must not pay for this at all.
        let language: LanguageRef = Arc::new(English);
        let tokens = language.tokenize("search engine offline");
        assert!(extract_phrases(&tokens, &language).is_empty());
    }

    #[test]
    fn single_bigram_is_not_a_phrase() {
        // Nothing to verify: one term is already an exact match.
        let language: LanguageRef = Arc::new(Japanese);
        let tokens = language.tokenize("日本");
        assert!(extract_phrases(&tokens, &language).is_empty());
    }

    #[test]
    fn contiguous_positions_verify() {
        let p = phrase(&["機械", "械学", "学習"]);
        let pos = positions(&[("機械", &[10]), ("械学", &[11]), ("学習", &[12])]);
        assert!(verify(&p, |t| pos.get(t).map(|v| v.as_slice())));
    }

    #[test]
    fn scattered_positions_do_not_verify() {
        // This is the false positive bigram indexing creates: every term is
        // present, none of them adjacent.
        let p = phrase(&["エン", "ンジ", "ジン"]);
        let pos = positions(&[("エン", &[3]), ("ンジ", &[40]), ("ジン", &[900])]);
        assert!(!verify(&p, |t| pos.get(t).map(|v| v.as_slice())));
    }

    #[test]
    fn missing_term_does_not_verify() {
        let p = phrase(&["機械", "械学", "学習"]);
        let pos = positions(&[("機械", &[10]), ("学習", &[12])]);
        assert!(!verify(&p, |t| pos.get(t).map(|v| v.as_slice())));
    }

    #[test]
    fn phrase_found_among_many_occurrences() {
        // The terms occur repeatedly, and exactly one alignment works.
        let p = phrase(&["機械", "械学", "学習"]);
        let pos = positions(&[
            ("機械", &[1, 50, 300]),
            ("械学", &[7, 51, 400]),
            ("学習", &[9, 52, 500]),
        ]);
        assert!(verify(&p, |t| pos.get(t).map(|v| v.as_slice())));
    }

    #[test]
    fn anchoring_does_not_underflow_at_offset_zero() {
        // The rarest term may be the last one, so the computed phrase start
        // can go negative. That must be skipped, not panic or wrap.
        let p = phrase(&["あい", "いう", "うえ"]);
        let pos = positions(&[("あい", &[0, 5, 9]), ("いう", &[1, 6]), ("うえ", &[2])]);
        assert!(verify(&p, |t| pos.get(t).map(|v| v.as_slice())));

        let pos = positions(&[("あい", &[8, 9]), ("いう", &[1, 6]), ("うえ", &[0])]);
        assert!(!verify(&p, |t| pos.get(t).map(|v| v.as_slice())));
    }

    #[test]
    fn repeated_term_within_a_phrase_verifies() {
        // 111 -> bigrams 11, 11 at offsets 0 and 1: the same term twice.
        let p = phrase(&["ああ", "ああ"]);
        let pos = positions(&[("ああ", &[4, 5])]);
        assert!(verify(&p, |t| pos.get(t).map(|v| v.as_slice())));

        let pos = positions(&[("ああ", &[4, 9])]);
        assert!(!verify(&p, |t| pos.get(t).map(|v| v.as_slice())));
    }

    #[test]
    fn cache_returns_a_stable_verdict_without_recomputing() {
        let p = phrase(&["機械", "械学"]);
        let pos = positions(&[("機械", &[1]), ("械学", &[2])]);
        let mut cache = VerificationCache::default();
        let mut calls = 0;

        for _ in 0..3 {
            let verdict = cache.verify(0, &p, "text", "doc-1", |t| {
                calls += 1;
                pos.get(t).map(|v| v.as_slice())
            });
            assert!(verdict);
        }
        assert_eq!(calls, 2, "verification must run once, not once per lookup");
    }

    #[test]
    fn cache_distinguishes_fields_and_documents() {
        let p = phrase(&["機械", "械学"]);
        let adjacent = positions(&[("機械", &[1]), ("械学", &[2])]);
        let scattered = positions(&[("機械", &[1]), ("械学", &[80])]);
        let mut cache = VerificationCache::default();

        assert!(cache.verify(0, &p, "title", "doc-1", |t| adjacent
            .get(t)
            .map(|v| v.as_slice())));
        assert!(!cache.verify(0, &p, "text", "doc-1", |t| scattered
            .get(t)
            .map(|v| v.as_slice())));
        assert!(!cache.verify(0, &p, "title", "doc-2", |t| scattered
            .get(t)
            .map(|v| v.as_slice())));
    }
}
