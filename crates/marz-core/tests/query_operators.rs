//! The query language, end to end.
//!
//! `query_parser`'s unit tests check that a query string produces the right AST.
//! These check that the AST produces the right documents, which is a different
//! failure: a correctly parsed boost that never reaches the scorer, or a presence
//! modifier that filters the wrong side. The assertions are on refs and relative
//! scores rather than magnitudes, for the reason given in `tests/golden.rs`.
//!
//! Replaces an earlier `manual.rs`, which printed its results and asserted
//! loosely. Its CJK cases are now in `tests/cjk.rs`, against an explained fixture.

use std::sync::Arc;

use marz_core::language::MultiLanguage;
use marz_core::languages::{Chinese, English, Japanese, Korean};
use marz_core::{Index, IndexBuilder, Language};

/// Two documents that share every term, differing only in which field holds it.
///
/// That is what makes field scoping and boosts observable: a query that ignored
/// the field would rank them equally.
fn index() -> Index {
    let mut builder = IndexBuilder::new(Arc::new(English) as Arc<dyn Language>);
    builder
        .ref_field("id")
        .field("title", 10.0)
        .field("body", 1.0);
    builder.add("a", 1.0, |name| match name {
        "title" => Some("marz search".to_string()),
        "body" => Some("a fast offline engine".to_string()),
        _ => None,
    });
    builder.add("b", 1.0, |name| match name {
        "title" => Some("offline engine".to_string()),
        "body" => Some("marz is fast and searchable".to_string()),
        _ => None,
    });
    builder.add("c", 1.0, |name| match name {
        "title" => Some("foobar".to_string()),
        "body" => Some("hello world".to_string()),
        _ => None,
    });
    builder.build()
}

fn refs(index: &Index, query: &str) -> Vec<String> {
    index
        .search(query)
        .unwrap_or_else(|e| panic!("query {query:?} failed: {e:?}"))
        .into_iter()
        .map(|r| r.ref_id)
        .collect()
}

#[test]
fn a_bare_term_matches_either_field() {
    let index = index();
    let mut got = refs(&index, "marz");
    got.sort();
    assert_eq!(got, ["a", "b"]);
}

#[test]
fn a_field_scope_restricts_to_that_field() {
    let index = index();
    assert_eq!(refs(&index, "title:marz"), ["a"]);
    assert_eq!(refs(&index, "body:marz"), ["b"]);
}

#[test]
fn the_field_boost_favours_the_boosted_field() {
    // title is boosted 10x, so document a — which has "marz" in its title —
    // must outrank b, which has it in a body.
    let index = index();
    assert_eq!(refs(&index, "marz"), ["a", "b"]);
}

#[test]
fn a_term_boost_reaches_the_scorer() {
    // Not just parsed: boosting the body occurrence hard enough must invert the
    // ranking the field boost produces on its own.
    let index = index();
    assert_eq!(refs(&index, "marz"), ["a", "b"]);
    assert_eq!(
        refs(&index, "title:marz body:marz^1000"),
        ["b", "a"],
        "a large term boost on the body did not outweigh the 10x title boost, so \
         the boost is being parsed but not applied"
    );
}

#[test]
fn a_required_term_must_be_present() {
    let index = index();
    // Both documents contain marz and engine, in some field.
    let mut got = refs(&index, "+marz +engine");
    got.sort();
    assert_eq!(got, ["a", "b"]);
    // c contains neither, and a required term it lacks excludes it even though
    // it would match nothing anyway — the assertion is that requiring a term
    // absent from every document yields nothing at all.
    assert!(refs(&index, "+marz +xylophone").is_empty());
}

#[test]
fn a_prohibited_term_excludes() {
    let index = index();
    // Every document containing marz also contains offline.
    assert!(refs(&index, "marz -offline").is_empty());
    // And a prohibition for something absent removes nothing.
    let mut got = refs(&index, "marz -xylophone");
    got.sort();
    assert_eq!(got, ["a", "b"]);
}

#[test]
fn a_prohibition_beats_a_requirement_for_the_same_term() {
    // Contradictory, and a user can type it. The result must be empty rather
    // than depending on which clause was evaluated last.
    let index = index();
    assert!(refs(&index, "+marz -marz").is_empty());
}

#[test]
fn wildcards_match_leading_trailing_and_both() {
    let index = index();
    assert_eq!(refs(&index, "foo*"), ["c"]);
    assert_eq!(refs(&index, "*bar"), ["c"]);
    assert_eq!(refs(&index, "*oob*"), ["c"]);
}

#[test]
fn a_wildcard_scoped_to_a_field_stays_scoped() {
    let index = index();
    assert_eq!(refs(&index, "title:foo*"), ["c"]);
    assert!(refs(&index, "body:foo*").is_empty());
}

#[test]
fn fuzzy_matching_tolerates_the_stated_edit_distance() {
    let index = index();
    // "helo" is one deletion from "hello".
    assert_eq!(refs(&index, "helo~1"), ["c"]);
    // Two edits away, so distance 1 must not reach it.
    assert!(refs(&index, "hllo~0").is_empty());
}

#[test]
fn stop_words_and_empty_queries_match_nothing() {
    let index = index();
    assert!(refs(&index, "").is_empty());
    assert!(refs(&index, "the").is_empty());
    // A stop word beside a real term must not suppress the real term.
    let mut got = refs(&index, "the marz");
    got.sort();
    assert_eq!(got, ["a", "b"]);
}

#[test]
fn stemming_collapses_inflections_on_both_sides() {
    let index = index();
    // "searchable" in b's body and "search" in a's title share a stem, so
    // either spelling finds both.
    let mut from_base = refs(&index, "search");
    let mut from_inflected = refs(&index, "searching");
    from_base.sort();
    from_inflected.sort();
    assert_eq!(from_base, from_inflected);
    assert!(from_base.contains(&"a".to_string()));
}

#[test]
fn an_unknown_field_is_an_error_not_an_empty_result() {
    // Silently returning nothing would make a typo in a search UI look like a
    // corpus with no matches.
    let index = index();
    let error = index.search("nosuchfield:marz").unwrap_err();
    assert!(
        format!("{error:?}").contains("nosuchfield"),
        "the error should name the offending field: {error:?}"
    );
}

#[test]
fn a_multi_language_index_serves_every_language_it_was_built_with() {
    // MultiLanguage dispatches per script within a single index, which is what a
    // documentation site with translated pages needs — one index, several
    // languages, no per-locale build.
    let language = Arc::new(MultiLanguage::new(vec![
        Arc::new(English) as Arc<dyn Language>,
        Arc::new(Chinese) as Arc<dyn Language>,
        Arc::new(Japanese) as Arc<dyn Language>,
        Arc::new(Korean) as Arc<dyn Language>,
    ]));

    let mut builder = IndexBuilder::new(language as Arc<dyn Language>);
    builder.ref_field("id").field("text", 1.0);
    builder.add("en", 1.0, |_| {
        Some("Rust is a systems programming language".to_string())
    });
    builder.add("zh", 1.0, |_| Some("中文搜索引擎的实现".to_string()));
    builder.add("ja", 1.0, |_| Some("日本語の検索エンジン".to_string()));
    builder.add("ko", 1.0, |_| Some("한국어 검색 엔진".to_string()));
    let index = builder.build();

    assert_eq!(refs(&index, "rust"), ["en"]);
    assert_eq!(refs(&index, "中文"), ["zh"]);
    assert_eq!(refs(&index, "日本語"), ["ja"]);
    assert_eq!(refs(&index, "한국어"), ["ko"]);

    // And a query in one language does not drag in documents in another. The
    // failure this guards is a shared bigram accidentally matching across
    // scripts, which is easy to introduce and hard to notice.
    for (query, expected) in [("rust", "en"), ("中文", "zh"), ("日本語", "ja")] {
        let got = refs(&index, query);
        assert_eq!(
            got.len(),
            1,
            "query {query:?} should match only {expected}, got {got:?}"
        );
    }
}

#[test]
fn a_query_spanning_two_scripts_matches_a_document_containing_both() {
    let language = Arc::new(MultiLanguage::new(vec![
        Arc::new(English) as Arc<dyn Language>,
        Arc::new(Japanese) as Arc<dyn Language>,
    ]));
    let mut builder = IndexBuilder::new(language as Arc<dyn Language>);
    builder.ref_field("id").field("text", 1.0);
    builder.add("both", 1.0, |_| {
        Some("Rustで検索エンジンを書く".to_string())
    });
    builder.add("latin", 1.0, |_| Some("Rust only here".to_string()));
    let index = builder.build();

    // Both terms present, so the mixed document outranks the one with only Rust.
    assert_eq!(refs(&index, "rust 検索"), ["both", "latin"]);
}
