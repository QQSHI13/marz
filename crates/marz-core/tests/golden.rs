//! Ranking parity against the lunr golden fixture.
//!
//! # What this test does and does not assert
//!
//! It asserts that Marz returns the **same documents in the same order** as
//! lunr for every fixture query. It deliberately does *not* assert that the
//! absolute scores match.
//!
//! lunr scores a document by taking an asymmetric cosine similarity between a
//! query vector and a precomputed field vector — the dot product divided by the
//! query vector's magnitude. Marz sums the same BM25 weights directly, with no
//! divisor (see [`marz_core::index`] for why the vectors are gone). The
//! divisor is constant across documents for a given query, so it rescales every
//! score by the same factor and cannot change their order; that is why ranking
//! survives and magnitude does not.
//!
//! Ranking is the observable behaviour: a search UI shows an ordered list. The
//! absolute float was never meaningful on its own, and pinning it to lunr's
//! normalization would freeze a design decision this rewrite reversed.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use marz_core::languages::English;
use marz_core::{Index, IndexBuilder, Language};
use serde::Deserialize;
use serde_json::Value;

/// One expected result from the fixture.
///
/// The fixture also records lunr's score for each result. It is not
/// deserialized, because Marz does not reproduce lunr's score magnitudes — see
/// the module docs.
#[derive(Debug, Deserialize)]
struct ExpectedResult {
    #[serde(rename = "ref")]
    ref_id: String,
}

fn fixtures() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn golden_index() -> Index {
    let documents_file = File::open(fixtures().join("documents.json")).unwrap();
    let documents: Vec<HashMap<String, Value>> = serde_json::from_reader(documents_file).unwrap();

    let language: Arc<dyn Language> = Arc::new(English);
    let mut builder = IndexBuilder::new(language);
    builder
        .ref_field("location")
        .field("title", 10.0)
        .field("text", 1.0);

    for doc in &documents {
        let boost = doc.get("_boost").and_then(|b| b.as_f64()).unwrap_or(1.0);
        builder.add(
            doc.get("location").unwrap().as_str().unwrap(),
            boost,
            |name| {
                doc.get(name)
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            },
        );
    }
    builder.build()
}

fn expected_results() -> HashMap<String, Vec<ExpectedResult>> {
    let queries_file = File::open(fixtures().join("queries.json")).unwrap();
    serde_json::from_reader(queries_file).unwrap()
}

#[test]
fn golden_english_ranking_parity() {
    let index = golden_index();

    for (query, expected) in expected_results() {
        let results = index.search(&query).unwrap();

        let got: Vec<&str> = results.iter().map(|r| r.ref_id.as_str()).collect();
        let want: Vec<&str> = expected.iter().map(|r| r.ref_id.as_str()).collect();

        assert_eq!(
            got, want,
            "query {query:?} returned a different ranking than lunr"
        );
    }
}

#[test]
fn golden_scores_are_well_formed() {
    // Ranking parity alone would still pass if every score were identical, so
    // check that the scores are real, usable numbers in descending order.
    let index = golden_index();

    for (query, expected) in expected_results() {
        if expected.is_empty() {
            continue;
        }
        // A purely negated query ("-marz") returns every non-excluded document
        // with score 0. That is correct: there is no positive term to score,
        // so there is no signal to rank by and nothing to assert about
        // magnitude. Only the exclusion itself is meaningful, and
        // `golden_english_ranking_parity` already covers it.
        if query.split_whitespace().all(|t| t.starts_with('-')) {
            continue;
        }
        let results = index.search(&query).unwrap();

        for result in &results {
            assert!(
                result.score.is_finite(),
                "query {query:?} ref {:?} scored {}",
                result.ref_id,
                result.score
            );
            assert!(
                result.score > 0.0,
                "query {query:?} ref {:?} scored {} — a matching document must \
                 score above zero",
                result.ref_id,
                result.score
            );
        }

        for pair in results.windows(2) {
            assert!(
                pair[0].score >= pair[1].score,
                "query {query:?} results are not sorted by descending score: \
                 {} ({}) before {} ({})",
                pair[0].ref_id,
                pair[0].score,
                pair[1].ref_id,
                pair[1].score
            );
        }
    }
}

#[test]
fn purely_negated_query_returns_the_complement_unscored() {
    // `golden_scores_are_well_formed` skips these, so assert the behaviour
    // directly: every document that does not contain the term, and no score.
    let index = golden_index();

    let excluded: Vec<String> = index
        .search("marz")
        .unwrap()
        .into_iter()
        .map(|r| r.ref_id)
        .collect();
    let negated = index.search("-marz").unwrap();

    assert!(!excluded.is_empty(), "fixture must contain the term 'marz'");
    assert!(!negated.is_empty(), "some documents must survive exclusion");
    for result in &negated {
        assert!(
            !excluded.contains(&result.ref_id),
            "{} matched 'marz' but survived '-marz'",
            result.ref_id
        );
        assert_eq!(
            result.score, 0.0,
            "a negated query has no positive term, so it carries no score signal"
        );
    }
}

#[test]
fn golden_scores_discriminate_between_documents() {
    // If scoring collapsed to a constant, ranking parity would be luck. Assert
    // that at least one multi-result query separates its documents by score.
    let index = golden_index();

    let discriminating = expected_results().keys().any(|query| {
        let results = index.search(query).unwrap();
        results.len() > 1 && results.first().unwrap().score > results.last().unwrap().score
    });

    assert!(
        discriminating,
        "no query produced distinct scores — scoring is not discriminating"
    );
}
