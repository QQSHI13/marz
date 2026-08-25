//! The binary format must preserve search behavior exactly.
//!
//! A size win is only worth having if the index still answers the same
//! questions. These tests compare a freshly built index against the same index
//! after a binary roundtrip, and require identical scores — not merely identical
//! ranking.
//!
//! Exact equality is the right assertion rather than an epsilon: every input to
//! BM25 that the format stores is an integer (term frequency, field length,
//! document frequency) or a value copied bit-for-bit (`k1`, `b`, boosts, all
//! `f64`). If the arithmetic diverges at all, something was lost or reordered,
//! and a tolerance would hide it.

use std::sync::Arc;

use marz_core::binary::{BinaryIndex, FormatError};
use marz_core::languages::{English, Japanese, Korean};
use marz_core::{Index, IndexBuilder, Language};

/// Documents whose refs contain slashes, whose fields differ in length, and
/// which include a document with an empty field.
const EN_DOCS: &[(&str, &str, &str)] = &[
    (
        "guide/install/index.html",
        "Installing Marz",
        "Install marz with cargo. The installer verifies the checksum.",
    ),
    (
        "guide/search/index.html",
        "Searching",
        "Search the index for terms. Searching is fast because the index is offline.",
    ),
    (
        "reference/api/index.html",
        "API reference",
        "The api exposes an index builder and a searcher.",
    ),
    ("empty/index.html", "Nothing here", ""),
];

const JA_DOCS: &[(&str, &str, &str)] = &[
    (
        "ja/1",
        "検索エンジン",
        "検索エンジンの仕組みについて説明します。全文検索は索引を使います。",
    ),
    (
        "ja/2",
        "エンジンオイル",
        "検索の話とエンジンオイルとジンの話をします。機械のエンジンです。",
    ),
    (
        "ja/3",
        "機械学習",
        "機械学習は人工知能の一分野です。統計的な学習理論に基づきます。",
    ),
];

const KO_DOCS: &[(&str, &str, &str)] = &[
    (
        "ko/1",
        "검색 엔진",
        "검색엔진의 원리를 설명합니다. 전문 검색은 색인을 사용합니다.",
    ),
    ("ko/2", "기계 학습", "기계학습은 인공지능의 한 분야입니다."),
];

fn build(language: Arc<dyn Language>, docs: &[(&str, &str, &str)]) -> Index {
    let mut builder = IndexBuilder::new(language);
    builder
        .ref_field("id")
        .field("title", 10.0)
        .field("body", 1.0);
    for (i, (id, title, body)) in docs.iter().enumerate() {
        let title = title.to_string();
        let body = body.to_string();
        // Vary the document boost so a roundtrip that dropped or misaligned
        // boosts would change the scores.
        builder.add(*id, 1.0 + i as f64 * 0.25, move |name| match name {
            "title" => Some(title.clone()),
            "body" => Some(body.clone()),
            _ => None,
        });
    }
    builder.build()
}

/// Compare every result of `query` between two indexes, field by field.
fn assert_same_results(original: &Index, roundtripped: &Index, query: &str) {
    let before = original.search(query).unwrap();
    let after = roundtripped.search(query).unwrap();

    assert_eq!(
        before.len(),
        after.len(),
        "query {query:?} returned {} results before and {} after",
        before.len(),
        after.len()
    );
    for (a, b) in before.iter().zip(after.iter()) {
        assert_eq!(a.ref_id, b.ref_id, "query {query:?}: ranking diverged");
        assert_eq!(
            a.score, b.score,
            "query {query:?} doc {}: score {} became {}",
            a.ref_id, a.score, b.score
        );

        // Match data drives highlighting, so positions must survive too.
        let mut terms_before: Vec<&String> = a.match_data.terms.keys().collect();
        let mut terms_after: Vec<&String> = b.match_data.terms.keys().collect();
        terms_before.sort();
        terms_after.sort();
        assert_eq!(
            terms_before, terms_after,
            "query {query:?}: match terms differ"
        );

        for (term, fields) in &a.match_data.terms {
            let other = &b.match_data.terms[term];
            for (field, positions) in fields {
                let mut expected = positions.clone();
                let mut actual = other[field].clone();
                expected.sort_unstable();
                actual.sort_unstable();
                assert_eq!(
                    expected, actual,
                    "query {query:?} term {term} field {field}: positions differ"
                );
            }
        }
    }
}

#[test]
fn english_search_is_identical_after_a_binary_roundtrip() {
    let index = build(Arc::new(English), EN_DOCS);
    let bytes = index.to_binary(true);
    let loaded = Index::from_binary(&bytes, Arc::new(English)).unwrap();

    for query in [
        "install",
        "installing",
        "search",
        "index",
        "api",
        "+search +index",
        "search -api",
        "title:search",
        "body:index",
        "sea*",
        "instal~1",
        "search^3 index",
        "search index api",
        "missingterm",
        "-api",
    ] {
        assert_same_results(&index, &loaded, query);
    }
}

#[test]
fn japanese_search_is_identical_after_a_binary_roundtrip() {
    // The case the format exists for: bigram postings, many positions, and
    // phrase verification that depends on those positions being exact.
    let index = build(Arc::new(Japanese), JA_DOCS);
    let bytes = index.to_binary(true);
    let loaded = Index::from_binary(&bytes, Arc::new(Japanese)).unwrap();

    for query in [
        "検索エンジン",
        "検索",
        "エンジン",
        "機械学習",
        "全文検索",
        "人工知能",
        "検索エンジン^2",
        "+検索 +索引",
        "検索 -機械",
        "title:検索エンジン",
    ] {
        assert_same_results(&index, &loaded, query);
    }
}

#[test]
fn korean_search_is_identical_after_a_binary_roundtrip() {
    let index = build(Arc::new(Korean), KO_DOCS);
    let bytes = index.to_binary(true);
    let loaded = Index::from_binary(&bytes, Arc::new(Korean)).unwrap();

    for query in ["검색엔진", "검색", "기계학습", "인공지능", "색인"] {
        assert_same_results(&index, &loaded, query);
    }
}

#[test]
fn phrase_boost_still_applies_after_a_roundtrip() {
    // Phrase verification reads positions out of the postings, so a roundtrip
    // that lost or shifted them would silently stop boosting — the scores would
    // still be plausible, just wrong. Assert the ranking the boost produces.
    let index = build(Arc::new(Japanese), JA_DOCS);
    let loaded = Index::from_binary(&index.to_binary(true), Arc::new(Japanese)).unwrap();

    let results = loaded.search("検索エンジン").unwrap();
    assert_eq!(
        results[0].ref_id, "ja/1",
        "the document with the literal phrase must still rank first"
    );
    assert!(results.len() > 1, "the scattered document must still match");
}

#[test]
fn positions_free_index_scores_the_same_but_loses_positions() {
    // Dropping positions is a size/feature tradeoff, not a scoring one: BM25
    // reads the stored term frequency, never the position count.
    let index = build(Arc::new(English), EN_DOCS);
    let with = index.to_binary(true);
    let without = index.to_binary(false);
    assert!(
        without.len() < with.len(),
        "positions-free index is {} bytes, not smaller than {}",
        without.len(),
        with.len()
    );

    let loaded = Index::from_binary(&without, Arc::new(English)).unwrap();
    for query in ["install", "search", "index", "api"] {
        let before = index.search(query).unwrap();
        let after = loaded.search(query).unwrap();
        assert_eq!(before.len(), after.len(), "query {query:?}");
        for (a, b) in before.iter().zip(after.iter()) {
            assert_eq!(a.ref_id, b.ref_id);
            assert_eq!(a.score, b.score, "query {query:?} doc {}", a.ref_id);
        }
        for result in &after {
            for fields in result.match_data.terms.values() {
                for positions in fields.values() {
                    assert!(positions.is_empty(), "positions should have been dropped");
                }
            }
        }
    }
}

#[test]
fn binary_is_substantially_smaller_than_json() {
    // The whole justification for the format. Measured on CJK text, where the
    // JSON overhead is worst.
    let index = build(Arc::new(Japanese), JA_DOCS);
    let json = index.to_json().len();
    let binary = index.to_binary(true).len();
    assert!(
        binary * 2 < json,
        "binary {binary} B should be under half of JSON {json} B"
    );
}

#[test]
fn document_refs_with_slashes_survive() {
    // The JSON format's `fieldName/docRef` key needed careful parsing to handle
    // these. The binary format interns references whole, so there is nothing to
    // parse — this test pins that.
    let index = build(Arc::new(English), EN_DOCS);
    let bytes = index.to_binary(true);
    let binary = BinaryIndex::open(&bytes).unwrap();

    let mut refs: Vec<&str> = (0..binary.doc_count() as u32)
        .map(|id| binary.doc_ref(id).unwrap())
        .collect();
    refs.sort_unstable();
    let mut expected: Vec<&str> = EN_DOCS.iter().map(|(id, _, _)| *id).collect();
    expected.sort_unstable();
    assert_eq!(refs, expected);
}

#[test]
fn reading_a_json_index_as_binary_fails_cleanly() {
    // A likely real mistake: a build script that swapped the two formats. It
    // must produce a clear error, not a garbled index.
    let index = build(Arc::new(English), EN_DOCS);
    let json = index.to_json();
    assert!(matches!(
        BinaryIndex::open(json.as_bytes()),
        Err(FormatError::BadMagic(_))
    ));
    assert!(Index::from_binary(json.as_bytes(), Arc::new(English)).is_err());
}

#[test]
fn empty_index_roundtrips() {
    let mut builder = IndexBuilder::new(Arc::new(English) as Arc<dyn Language>);
    builder.ref_field("id").field("body", 1.0);
    let index = builder.build();

    let bytes = index.to_binary(true);
    let loaded = Index::from_binary(&bytes, Arc::new(English)).unwrap();
    assert_eq!(loaded.document_count(), 0);
    assert_eq!(loaded.term_count(), 0);
    assert!(loaded.search("anything").unwrap().is_empty());
}

#[test]
fn truncated_binary_index_is_rejected_not_misread() {
    let index = build(Arc::new(Japanese), JA_DOCS);
    let bytes = index.to_binary(true);

    // Every truncation either fails to load, or loads and answers queries
    // without panicking. What must never happen is a panic or a hang.
    for length in (0..bytes.len()).step_by(3) {
        if let Ok(loaded) = Index::from_binary(&bytes[..length], Arc::new(Japanese)) {
            let _ = loaded.search("検索エンジン");
        }
    }
}
