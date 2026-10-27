//! Score finiteness, operator parsing around exotic spaces, and wildcard
//! expansion against stemmed terms.
//!
//! Three classes of bug that produce plausible-looking wrong answers rather than
//! errors, so each needs a test that pins the *behaviour* and not merely the
//! absence of a panic:
//!
//! * **Non-finite scores** (`inf`, `NaN`) sort unpredictably and compare wrongly
//!   in every caller. Finite inputs must not be able to produce them.
//! * **Spaces that fold to ASCII space** split terms at tokenization but not at
//!   lexing, which silently disables `+`, `-`, `field:` and `*`.
//! * **Wildcards over stemmed terms**: the index holds stems, the query holds
//!   surface text, and the two disagree at the edges.

use std::sync::Arc;

use marz_core::languages::{English, Japanese};
use marz_core::{Index, IndexBuilder, Language};

// ---------------------------------------------------------------------------
// #2: finite arithmetic
// ---------------------------------------------------------------------------

#[test]
fn a_long_repeated_cjk_phrase_scores_finitely() {
    // 1,030 repetitions of a three-token phrase produce 1,029 verified phrases,
    // so the multiplicative phrase boost is PHRASE_BOOST^1029 — which is `inf`
    // in f64. The score must saturate instead.
    let mut builder = IndexBuilder::new(Arc::new(Japanese) as Arc<dyn Language>);
    builder.ref_field("id").field("body", 1.0);
    let text = "検索エンジン".repeat(1_030);
    builder.add("ja", 1.0, |name| match name {
        "body" => Some(text.clone()),
        _ => None,
    });
    let index = builder.build();

    let results = index.search("検索エンジン").unwrap();
    assert!(!results.is_empty(), "the document must match its own text");
    for result in &results {
        assert!(
            result.score.is_finite(),
            "{} scored {} for a long repeated phrase",
            result.ref_id,
            result.score
        );
    }
}

#[test]
fn a_long_query_of_repeated_cjk_runs_scores_finitely() {
    // 4,095 characters of `あああ` runs split into thousands of bigrams, each
    // contributing to the same document. Summing them used to produce `NaN` in a
    // legal zero-boost field.
    let mut builder = IndexBuilder::new(Arc::new(Japanese) as Arc<dyn Language>);
    builder.ref_field("id").field("body", 1.0);
    builder.add("ja", 1.0, |name| match name {
        "body" => Some("ああああああああああ".to_string()),
        _ => None,
    });
    let index = builder.build();

    let query = "あああ ".repeat(820);
    for suffix in ["", "^0"] {
        let results = index
            .search(&format!("{query}{suffix}"))
            .expect("a long query must parse");
        for result in &results {
            assert!(
                result.score.is_finite(),
                "score {} for query length {}",
                result.score,
                query.len()
            );
        }
    }
}

#[test]
fn repeated_huge_boosts_score_finitely() {
    // Eight clauses at f64::MAX-adjacent boosts: each is finite on its own, so
    // this is not a parse error, but their sum was `inf`.
    let mut builder = IndexBuilder::new(Arc::new(English) as Arc<dyn Language>);
    builder.ref_field("id").field("body", 1.0);
    builder.add("a", 1.0, |name| match name {
        "body" => Some("hello hello hello hello".to_string()),
        _ => None,
    });
    builder.add("b", 1.0, |name| match name {
        "body" => Some("hello".to_string()),
        _ => None,
    });
    let index = builder.build();

    let huge = format!("{}", 1.7e308_f64);
    let query = std::iter::repeat_n(format!("hello^{huge}"), 8)
        .collect::<Vec<_>>()
        .join(" ");
    let results = index.search(&query).expect("huge boosts are finite");
    assert!(!results.is_empty());
    for result in &results {
        assert!(
            result.score.is_finite() && result.score > 0.0,
            "{} scored {} for repeated large boosts",
            result.ref_id,
            result.score
        );
    }
}

#[test]
fn a_zero_boost_is_zero_not_nan() {
    // `0 × inf` is `NaN`, which is why the multiplication short-circuits on zero
    // rather than relying on the arithmetic. A zero-boosted clause must match
    // (presence still applies) and contribute exactly nothing.
    let mut builder = IndexBuilder::new(Arc::new(English) as Arc<dyn Language>);
    builder.ref_field("id").field("body", 0.0);
    builder.add("a", 1.0, |name| match name {
        "body" => Some("keyboard keyboard keyboard".to_string()),
        _ => None,
    });
    let index = builder.build();

    let results = index.search("keyboard^0").unwrap();
    assert_eq!(results.len(), 1, "a zero boost still matches");
    assert_eq!(results[0].score, 0.0, "a zero boost contributes nothing");

    // And a zero-boosted field with a real clause boost is still zero.
    let results = index.search(&format!("keyboard^{}", f64::MAX)).unwrap();
    assert!(results.iter().all(|r| r.score == 0.0));
}

// ---------------------------------------------------------------------------
// #3: exotic spaces must not disable query operators
// ---------------------------------------------------------------------------

/// Three documents: one has `foo` only, one has `bar` only, one has both.
fn operator_fixture() -> Index {
    let mut builder = IndexBuilder::new(Arc::new(English) as Arc<dyn Language>);
    builder.ref_field("id").field("body", 1.0);
    builder.add("foo", 1.0, |name| match name {
        "body" => Some("foo".to_string()),
        _ => None,
    });
    builder.add("bar", 1.0, |name| match name {
        "body" => Some("bar".to_string()),
        _ => None,
    });
    builder.add("both", 1.0, |name| match name {
        "body" => Some("foo bar".to_string()),
        _ => None,
    });
    builder.build()
}

fn refs(index: &Index, query: &str) -> Vec<String> {
    let mut out: Vec<String> = index
        .search(query)
        .unwrap_or_else(|e| panic!("query {query:?} failed: {e:?}"))
        .into_iter()
        .map(|r| r.ref_id)
        .collect();
    out.sort();
    out
}

#[test]
fn an_ideographic_space_keeps_the_prohibition_operator() {
    // The bug: `foo -bar` excludes `bar`, but `foo　-bar` (U+3000) lexed as one
    // term and returned all three documents — including the prohibited one.
    let index = operator_fixture();
    assert_eq!(refs(&index, "foo -bar"), ["foo"]);
    for space in ['\u{3000}', '\u{2003}', '\u{202F}', '\u{205F}', '\u{2000}'] {
        let query = format!("foo{space}-bar");
        assert_eq!(
            refs(&index, &query),
            ["foo"],
            "prohibition must survive a U+{:04X} separator",
            space as u32
        );
    }
}

#[test]
fn an_ideographic_space_keeps_the_required_operator() {
    let index = operator_fixture();
    assert_eq!(refs(&index, "+foo +bar"), ["both"]);
    let space = '\u{3000}';
    let query = format!("+foo{space}+bar");
    assert_eq!(
        refs(&index, &query),
        ["both"],
        "both requirements must survive an ideographic space"
    );
}

#[test]
fn an_ideographic_space_keeps_field_scope_and_wildcards() {
    let mut builder = IndexBuilder::new(Arc::new(English) as Arc<dyn Language>);
    builder
        .ref_field("id")
        .field("title", 1.0)
        .field("body", 1.0);
    builder.add("a", 1.0, |name| match name {
        "title" => Some("keyboard".to_string()),
        "body" => Some("mouse".to_string()),
        _ => None,
    });
    builder.add("b", 1.0, |name| match name {
        "title" => Some("mouse".to_string()),
        "body" => Some("keyboard".to_string()),
        _ => None,
    });
    let index = builder.build();

    assert_eq!(refs(&index, "title:keyboard"), ["a"]);
    assert_eq!(
        refs(&index, "title:\u{3000}keyboard"),
        ["a"],
        "field scope must survive an ideographic space after the colon"
    );
    // A wildcard after an ideographic space must still expand.
    assert_eq!(refs(&index, "x\u{3000}key*"), ["a", "b"]);
}

// ---------------------------------------------------------------------------
// #5 / #6: wildcards over stemmed terms
// ---------------------------------------------------------------------------

/// Documents chosen so the stemmer's two failure modes are both observable.
fn stem_fixture() -> Index {
    let mut builder = IndexBuilder::new(Arc::new(English) as Arc<dyn Language>);
    builder.ref_field("id").field("body", 1.0);
    for (id, word) in [
        // indexed as `abyss`; surface prefix `aby` stems to `abi`
        ("abysses", "abysses"),
        // indexed as `abacin`; surface prefix `abas` stems to `aba`
        ("abacinate", "abacinate"),
        // indexed as `secur`; `secur*ity` must not match it
        ("secure", "secure"),
        // indexed as `searchabl`; surface prefix `searchable` is longer
        ("searchable", "searchable"),
    ] {
        builder.add(id, 1.0, move |name| match name {
            "body" => Some(word.to_string()),
            _ => None,
        });
    }
    builder.build()
}

#[test]
fn an_interior_wildcard_is_not_stemmed_into_a_broader_pattern() {
    // Porter treats `*` as an ordinary character: `secur*ity` stemmed to
    // `secur*`, which then matched every term starting with `secur` — including
    // a document holding only `secure`. An interior star must be matched as
    // written, against the stored stems, exactly like lunr.
    let index = stem_fixture();
    assert!(
        refs(&index, "secur*ity").is_empty(),
        "secur*ity must not match `secure`: the body has an interior star"
    );
    // And the interior glob still matches an indexed term that fits it. The word
    // is chosen so the stemmer leaves it alone — the glob runs against stored
    // stems, so a stemmed word would test the stemmer rather than the glob.
    let mut builder = IndexBuilder::new(Arc::new(English) as Arc<dyn Language>);
    builder.ref_field("id").field("body", 1.0);
    builder.add("x", 1.0, |name| match name {
        "body" => Some("abcz".to_string()),
        _ => None,
    });
    let index = builder.build();
    assert_eq!(
        refs(&index, "abc*z"),
        ["x"],
        "an interior glob still matches"
    );
    assert_eq!(
        refs(&index, "ab*z"),
        ["x"],
        "the glob spans the rest of the term"
    );
    assert!(
        refs(&index, "abx*z").is_empty(),
        "literal text in the glob is matched literally"
    );
}

#[test]
fn a_surface_prefix_matches_terms_whose_stem_is_shorter() {
    // `abysses` indexes as `abyss`, but stemming the prefix `aby` gives `abi`, so
    // expanding only the stemmed pattern found nothing. The raw pattern must be
    // tried as well.
    let index = stem_fixture();
    assert_eq!(refs(&index, "aby*"), ["abysses"], "aby* must find abysses");
    assert_eq!(
        refs(&index, "abys*"),
        ["abysses"],
        "abys* must find abysses"
    );
    assert_eq!(
        refs(&index, "abyss*"),
        ["abysses"],
        "the full stem must still find it"
    );
}

#[test]
fn a_surface_prefix_does_not_overmatch_through_stemming() {
    // The mirror case: `abas` stems to `aba`, which is a prefix of `abacin`, so
    // expanding the stemmed pattern found `abacinate` for a query that does not
    // match its text. Only a raw-pattern expansion is correct here.
    let index = stem_fixture();
    assert!(
        refs(&index, "abas*").is_empty(),
        "abas* must not find `abacinate`"
    );
}

#[test]
fn a_surface_prefix_that_outruns_its_stem_still_matches() {
    // `searchable` indexes as `searchabl`, which is *shorter*. The raw pattern
    // `searchable*` cannot match it, and dropping the stemmed lookup entirely
    // would trade one bug for another.
    let index = stem_fixture();
    assert_eq!(
        refs(&index, "searchable*"),
        ["searchable"],
        "a prefix longer than its own stem must still match"
    );
}

#[test]
fn stars_only_and_leading_wildcards_still_work() {
    let index = stem_fixture();
    let mut all = refs(&index, "*");
    all.sort();
    assert_eq!(all.len(), 4, "a bare star matches every document: {all:?}");
    assert_eq!(
        refs(&index, "*abl").len(),
        1,
        "a leading star still matches"
    );
}

// ---------------------------------------------------------------------------
// #10: `limit` must not change which results come back
// ---------------------------------------------------------------------------

fn broad_fixture() -> Index {
    let mut builder = IndexBuilder::new(Arc::new(English) as Arc<dyn Language>);
    builder.ref_field("id").field("body", 1.0);
    for i in 0..200 {
        builder.add(format!("doc-{i:03}"), 1.0, move |name| match name {
            "body" => Some(format!("common term{i} common")),
            _ => None,
        });
    }
    builder.build()
}

#[test]
fn a_limited_search_returns_the_same_top_k_as_an_unlimited_one() {
    // Building match data used to happen for every hit before `limit` was
    // applied, so a broad wildcard over a real corpus spent seconds and hundreds
    // of megabytes building metadata for results the caller never saw. The
    // truncation moved into core; the visible results must be unchanged.
    let index = broad_fixture();
    for query in ["common", "com*", "*", "term*, common", "common^2 -term7"] {
        let all = index.search(query).expect("query");
        assert!(all.len() > 10, "fixture must be broad enough: {query}");
        for limit in [0usize, 1, 5, 20, all.len(), all.len() + 10] {
            let limited = index
                .search_limited(query, Some(limit))
                .expect("limited query");
            assert_eq!(
                limited.len(),
                limit.min(all.len()),
                "limit {limit} produced {} results for {query:?}",
                limited.len()
            );
            for (short, full) in limited.iter().zip(all.iter()) {
                assert_eq!(short.ref_id, full.ref_id, "order changed by limit");
                assert_eq!(short.score, full.score, "score changed by limit");
                assert_eq!(
                    short.match_data.terms, full.match_data.terms,
                    "match data changed by limit for {}",
                    short.ref_id
                );
            }
        }
    }
}

#[test]
fn a_negated_query_honours_a_limit() {
    let index = broad_fixture();
    let all = index.search("-term7").expect("query");
    let limited = index.search_limited("-term7", Some(3)).expect("query");
    assert_eq!(limited.len(), 3);
    for (short, full) in limited.iter().zip(all.iter()) {
        assert_eq!(short.ref_id, full.ref_id);
    }
}

// ---------------------------------------------------------------------------
// #14: wildcard scores must be reproducible across reloads
// ---------------------------------------------------------------------------

#[test]
fn wildcard_scores_are_identical_across_repeated_reloads() {
    // Trie edges live in a `HashMap`, so expansion order used to vary between
    // loads and float contributions were summed in a different order each time:
    // 48 score mismatches across 20 reloads, differing by a few ULPs — enough to
    // flip a near-tied ranking. Expansion order is canonicalized now, so the
    // same bytes must produce bit-identical scores every time.
    let mut builder = IndexBuilder::new(Arc::new(English) as Arc<dyn Language>);
    builder.ref_field("id").field("body", 1.0);
    for i in 0..60 {
        builder.add(format!("d{i:02}"), 1.0, move |name| match name {
            "body" => Some(format!(
                "ab{i} abc{i} abd{i} word{i} medieval notice keyboard"
            )),
            _ => None,
        });
    }
    let bytes = builder.build().to_binary(true).unwrap();

    let queries = [
        "*",
        "**",
        "ab*",
        "*ard",
        "a*c",
        "med*val",
        "keyboard~1",
        "notic*",
    ];
    // The baseline, then nineteen more full loads.
    let baseline: Vec<Vec<(String, f64)>> = {
        let index = Index::from_binary(&bytes, Arc::new(English)).unwrap();
        queries
            .iter()
            .map(|q| {
                index
                    .search(q)
                    .unwrap()
                    .into_iter()
                    .map(|r| (r.ref_id, r.score))
                    .collect()
            })
            .collect()
    };

    for reload in 0..19 {
        let index = Index::from_binary(&bytes, Arc::new(English)).unwrap();
        for (i, query) in queries.iter().enumerate() {
            let got: Vec<(String, f64)> = index
                .search(query)
                .unwrap()
                .into_iter()
                .map(|r| (r.ref_id, r.score))
                .collect();
            assert_eq!(
                got.len(),
                baseline[i].len(),
                "reload {reload} query {query:?} returned a different number of hits"
            );
            for (a, b) in got.iter().zip(baseline[i].iter()) {
                assert_eq!(a.0, b.0, "reload {reload} query {query:?} order flipped");
                // Bit-identical, not merely close: a ULP difference is exactly
                // what made near-tied rankings flip before.
                assert_eq!(
                    a.1.to_bits(),
                    b.1.to_bits(),
                    "reload {reload} query {query:?} scored {} as {} instead of {}",
                    a.0,
                    a.1,
                    b.1
                );
            }
        }
    }
}
