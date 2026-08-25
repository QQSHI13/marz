//! CJK phrase verification: does adjacency actually improve ranking?
//!
//! These tests use controlled documents rather than a real corpus, because on
//! real text a title-field boost and term frequency already separate good hits
//! from bad ones, which masks the effect being tested. Here the decoy is
//! constructed to be indistinguishable from the target under bag-of-bigrams
//! scoring: it contains every query bigram, at a comparable frequency, in a
//! field of comparable length — and differs only in that the bigrams are not
//! adjacent.

use std::sync::Arc;

use marz_core::languages::{English, Japanese, Korean};
use marz_core::{IndexBuilder, Language};

fn index(language: Arc<dyn Language>, docs: &[(&str, &str)]) -> marz_core::Index {
    let mut builder = IndexBuilder::new(language);
    builder.ref_field("id").field("text", 1.0);
    for (id, text) in docs {
        let text = text.to_string();
        builder.add(*id, 1.0, move |name| match name {
            "text" => Some(text.clone()),
            _ => None,
        });
    }
    builder.build()
}

fn ranking(index: &marz_core::Index, query: &str) -> Vec<String> {
    index
        .search(query)
        .unwrap()
        .into_iter()
        .map(|r| r.ref_id)
        .collect()
}

#[test]
fn japanese_phrase_outranks_scattered_bigrams() {
    // "検索エンジン" (search engine) tokenizes to 検索 / エン ンジ ジン.
    //
    // The decoy contains all four bigrams: エンジン appears as part of a
    // mechanical engine, ジン as gin. Under bag-of-bigrams scoring it looks
    // just like the real hit.
    let index = index(
        Arc::new(Japanese),
        &[
            ("phrase", "検索エンジンの仕組みについて説明します。"),
            ("scattered", "検索の話とエンジンオイルとジンの話をします。"),
        ],
    );

    let results = index.search("検索エンジン").unwrap();
    assert_eq!(results.len(), 2, "both documents must still match");
    assert_eq!(
        results[0].ref_id, "phrase",
        "the document containing the literal phrase must rank first"
    );
    assert!(
        results[0].score > results[1].score,
        "phrase {} must score above scattered {}",
        results[0].score,
        results[1].score
    );
}

#[test]
fn korean_phrase_outranks_scattered_bigrams() {
    // 기계학습 -> 기계 / 계학 / 학습.
    let index = index(
        Arc::new(Korean),
        &[
            ("phrase", "기계학습은 인공지능의 한 분야입니다."),
            ("scattered", "기계 부품과 통계학 학습 방법을 설명합니다."),
        ],
    );

    let results = index.search("기계학습").unwrap();
    assert_eq!(results[0].ref_id, "phrase");
}

#[test]
fn scattered_bigrams_still_match() {
    // Verification boosts, it does not filter. A document with the bigrams
    // scattered is a weaker match, not a non-match — dropping it would trade
    // away the recall that made bigram indexing worth choosing.
    let index = index(
        Arc::new(Japanese),
        &[("scattered", "検索の話とエンジンオイルとジンの話をします。")],
    );
    assert_eq!(ranking(&index, "検索エンジン"), ["scattered"]);
}

#[test]
fn phrase_boost_never_applies_to_a_word_tokenized_language() {
    // English produces no phrases, so no score may ever be multiplied. Assert
    // that directly rather than pinning a ranking: comparing a two-term query
    // against its single-term parts shows whether any extra factor crept in.
    let docs = &[
        ("a", "the search engine indexes documents offline"),
        ("b", "a search for a good engine"),
        ("c", "documents about offline storage"),
    ];
    let index = index(Arc::new(English), docs);

    // With no phrase boost, a disjunctive query's score is exactly the sum of
    // its terms' scores. Any phrase multiplier would break that identity.
    let score_of = |query: &str, id: &str| -> f64 {
        index
            .search(query)
            .unwrap()
            .into_iter()
            .find(|r| r.ref_id == id)
            .map(|r| r.score)
            .unwrap_or(0.0)
    };

    for id in ["a", "b"] {
        let combined = score_of("search engine", id);
        let parts = score_of("search", id) + score_of("engine", id);
        assert!(
            (combined - parts).abs() < 1e-9,
            "{id}: 'search engine' scored {combined} but 'search' + 'engine' = \
             {parts} — a phrase boost leaked into a word-tokenized language"
        );
    }
}

#[test]
fn single_bigram_query_is_unaffected() {
    // 日本 is one bigram, so there is no phrase and nothing to verify. The two
    // documents must rank by term frequency and field length alone.
    let index = index(
        Arc::new(Japanese),
        &[
            ("many", "日本の日本語と日本人について。"),
            ("one", "日本について少し。"),
        ],
    );
    let results = index.search("日本").unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].ref_id, "many");
}

#[test]
fn phrase_across_script_boundary_verifies_per_script_run() {
    // 検索エンジン splits at the Han/Katakana boundary, so 検索 is a standalone
    // bigram and only エン ンジ ジン form a phrase. A document with the
    // katakana word intact should still beat one that scatters it, even though
    // the Han part is common to both.
    let index = index(
        Arc::new(Japanese),
        &[
            ("intact", "検索エンジンを使う。"),
            ("split", "検索します。エンオイルとンジとジンはこちら。"),
        ],
    );
    let results = index.search("検索エンジン").unwrap();
    assert_eq!(results[0].ref_id, "intact");
}

#[test]
fn repeated_phrase_scores_above_single_occurrence() {
    // The boost is per phrase-match, not per occurrence, but term frequency
    // still applies, so more occurrences must still rank higher.
    let index = index(
        Arc::new(Japanese),
        &[
            ("twice", "機械学習と機械学習の応用。"),
            ("once", "機械学習の応用について詳しく述べる長い文章です。"),
        ],
    );
    let results = index.search("機械学習").unwrap();
    assert_eq!(results[0].ref_id, "twice");
}

#[test]
fn wildcard_query_skips_phrase_verification_without_panicking() {
    // A wildcard disables the pipeline, so no phrases are extracted. This must
    // degrade to plain bigram matching rather than misbehave.
    let index = index(Arc::new(Japanese), &[("a", "検索エンジンの仕組み。")]);
    assert_eq!(ranking(&index, "検索*"), ["a"]);
    assert!(!ranking(&index, "*").is_empty());
}

#[test]
fn phrase_verification_respects_field_scope() {
    // The phrase must be verified within a single field. Bigrams split across
    // title and body are not a phrase, even though positions could align by
    // coincidence.
    let language: Arc<dyn Language> = Arc::new(Japanese);
    let mut builder = IndexBuilder::new(language);
    builder
        .ref_field("id")
        .field("title", 1.0)
        .field("text", 1.0);
    builder.add("split", 1.0, |name| match name {
        // "エン" ends the title, "ンジジン" starts the body: adjacent only if
        // fields were concatenated, which they are not.
        "title" => Some("エン".to_string()),
        "text" => Some("ンジジン".to_string()),
        _ => None,
    });
    builder.add("intact", 1.0, |name| match name {
        "title" => Some("エンジン".to_string()),
        "text" => Some("エンジンについて".to_string()),
        _ => None,
    });
    let index = builder.build();

    let results = index.search("エンジン").unwrap();
    assert_eq!(
        results[0].ref_id, "intact",
        "a phrase spanning two fields must not verify"
    );
}
