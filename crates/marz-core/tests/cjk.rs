//! CJK search behaviour, against a hand-specified fixture.
//!
//! # Why this fixture is not generated
//!
//! `tests/fixtures/queries.json` is generated from lunr.js, because English
//! ranking has a reference implementation to be checked against. CJK has none:
//! lunr splits on whitespace, so an unspaced run of Han is one enormous token and
//! a search for a word inside it finds nothing. That is the gap Marz exists to
//! close, which means there is nothing to generate the expectations *from*.
//!
//! So `tests/fixtures/cjk.json` is written by hand, and every case carries a
//! `why` explaining what it is for. That is the load-bearing part. A fixture of
//! bare query/result pairs recorded from the current implementation asserts only
//! that the code still does what it did, and silently re-blesses a regression the
//! moment someone regenerates it. A case with a stated reason can be argued with.
//!
//! # What the shapes mean
//!
//! - `expect` pins refs **in rank order**. Use it when the ordering is the
//!   behaviour — phrase matches above scattered ones, title hits above body hits.
//! - `contains` asserts presence without pinning order, for cases where the
//!   ranking between the named documents is not what is being tested.
//! - `expect_empty` asserts nothing matched, which is what stops bigram
//!   tokenization from quietly making everything match everything.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use marz_core::languages::{Chinese, English, Japanese, Korean};
use marz_core::{Index, IndexBuilder, Language};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Fixture {
    languages: HashMap<String, LanguageCases>,
}

#[derive(Debug, Deserialize)]
struct LanguageCases {
    documents: Vec<Document>,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Document {
    id: String,
    title: String,
    body: String,
}

#[derive(Debug, Deserialize)]
struct Case {
    query: String,
    /// Refs in required rank order.
    #[serde(default)]
    expect: Option<Vec<String>>,
    /// Refs that must appear, in any order.
    #[serde(default)]
    contains: Option<Vec<String>>,
    #[serde(default)]
    expect_empty: bool,
    /// Why this case exists. Read by `every_case_explains_itself`, which is the
    /// mechanism that keeps the fixture from decaying into recorded output.
    why: String,
}

fn fixture() -> Fixture {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/cjk.json");
    let file = File::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    serde_json::from_reader(file).expect("parse cjk.json")
}

fn language_for(code: &str) -> Arc<dyn Language> {
    match code {
        "zh" => Arc::new(Chinese),
        "ja" => Arc::new(Japanese),
        "ko" => Arc::new(Korean),
        "en" => Arc::new(English),
        other => panic!("fixture names an unknown language {other:?}"),
    }
}

fn build(code: &str, documents: &[Document]) -> Index {
    let mut builder = IndexBuilder::new(language_for(code));
    builder
        .ref_field("id")
        .field("title", 10.0)
        .field("body", 1.0);
    for doc in documents {
        let title = doc.title.clone();
        let body = doc.body.clone();
        builder.add(doc.id.clone(), 1.0, move |name| match name {
            "title" => Some(title.clone()),
            "body" => Some(body.clone()),
            _ => None,
        });
    }
    builder.build()
}

fn refs(index: &Index, query: &str) -> Vec<String> {
    index
        .search(query)
        .unwrap_or_else(|e| panic!("query {query:?} failed to parse: {e:?}"))
        .into_iter()
        .map(|r| r.ref_id)
        .collect()
}

#[test]
fn cjk_fixture_expectations_hold() {
    let fixture = fixture();
    let mut checked = 0usize;

    for (code, cases) in &fixture.languages {
        let index = build(code, &cases.documents);

        for case in &cases.cases {
            let got = refs(&index, &case.query);
            let context = || format!("[{code}] {:?} — {}", case.query, case.why);

            if case.expect_empty {
                assert!(
                    got.is_empty(),
                    "{}\n  expected no results, got {got:?}",
                    context()
                );
            }
            if let Some(expected) = &case.expect {
                assert_eq!(&got, expected, "{}\n  ranking differs", context());
            }
            if let Some(required) = &case.contains {
                for want in required {
                    assert!(
                        got.contains(want),
                        "{}\n  {want:?} missing from {got:?}",
                        context()
                    );
                }
            }
            checked += 1;
        }
    }

    // A fixture that failed to deserialize into anything would otherwise pass.
    assert!(checked >= 18, "only {checked} cases ran");
}

#[test]
fn a_phrase_match_outranks_a_partial_one_by_a_wide_margin() {
    // Phrase verification boosts, it does not filter. A document containing 機械
    // but not 機械学習 still matches a query for 機械学習, because excluding it
    // would make CJK queries behave as AND while the same query in English
    // behaves as OR — a worse surprise than a low-ranked extra hit.
    //
    // What makes that acceptable is the size of the gap, so the gap is the
    // assertion. A margin of a few percent would mean the partial match competes
    // for the top of the list; two orders of magnitude means it sits where nobody
    // looks unless nothing better exists.
    let fixture = fixture();
    let ja = &fixture.languages["ja"];
    let index = build("ja", &ja.documents);

    let results = index.search("機械学習").unwrap();
    let phrase = results
        .iter()
        .find(|r| r.ref_id == "ja/ml")
        .expect("ja/ml contains the phrase");
    let partial = results
        .iter()
        .find(|r| r.ref_id == "ja/scattered")
        .expect("ja/scattered contains 機械 only, and should still match");

    assert!(
        phrase.score > partial.score * 20.0,
        "phrase match scored {} and partial match {}, a margin of only {:.1}x — \
         too narrow for the partial hit to be harmless",
        phrase.score,
        partial.score,
        phrase.score / partial.score
    );
}

#[test]
fn every_case_explains_itself() {
    // The fixture's value is in the reasons, not the pairs. An unexplained case
    // is indistinguishable from recorded output, and recorded output cannot
    // detect a regression — it *is* whatever the code last did.
    for (code, cases) in &fixture().languages {
        for case in &cases.cases {
            assert!(
                case.why.len() > 30,
                "[{code}] {:?} has no real explanation: {:?}",
                case.query,
                case.why
            );
            assert!(
                case.expect.is_some() || case.contains.is_some() || case.expect_empty,
                "[{code}] {:?} asserts nothing",
                case.query
            );
        }
    }
}

#[test]
fn the_fixture_covers_every_cjk_language() {
    let fixture = fixture();
    for code in ["zh", "ja", "ko"] {
        let cases = fixture
            .languages
            .get(code)
            .unwrap_or_else(|| panic!("fixture has no cases for {code}"));
        assert!(
            cases.documents.len() >= 3,
            "{code} has only {} documents",
            cases.documents.len()
        );
        assert!(
            cases.cases.iter().any(|c| c.expect_empty),
            "{code} has no negative case; without one, a tokenizer that matched \
             everything would pass"
        );
    }
}

#[test]
fn cjk_fixture_survives_a_binary_roundtrip() {
    // The expectations must hold through the format that actually ships, not
    // only through the in-memory index the builder produces.
    let fixture = fixture();

    for (code, cases) in &fixture.languages {
        let original = build(code, &cases.documents);
        let bytes = original.to_binary(true);
        let loaded = Index::from_binary(&bytes, language_for(code))
            .unwrap_or_else(|e| panic!("[{code}] roundtrip failed: {e:?}"));

        for case in &cases.cases {
            let before = original.search(&case.query).unwrap();
            let after = loaded.search(&case.query).unwrap();
            assert_eq!(
                before.len(),
                after.len(),
                "[{code}] {:?}: {} results became {}",
                case.query,
                before.len(),
                after.len()
            );
            for (a, b) in before.iter().zip(after.iter()) {
                assert_eq!(
                    a.ref_id, b.ref_id,
                    "[{code}] {:?}: ranking diverged",
                    case.query
                );
                assert_eq!(
                    a.score, b.score,
                    "[{code}] {:?} doc {}: score {} became {}",
                    case.query, a.ref_id, a.score, b.score
                );
            }
        }
    }
}

#[test]
fn a_positions_free_index_still_answers_the_same_documents() {
    // Dropping positions is the size/functionality trade the format offers. It
    // costs highlighting and phrase verification, so ranking may change — but a
    // query must not start returning documents that do not contain its terms.
    let fixture = fixture();

    for (code, cases) in &fixture.languages {
        let full = build(code, &cases.documents);
        let bytes = full.to_binary(false);
        let lean = Index::from_binary(&bytes, language_for(code)).unwrap();

        for case in &cases.cases {
            let with = refs(&full, &case.query);
            let without = refs(&lean, &case.query);

            // Phrase verification can only ever remove documents, so the
            // positions-bearing index is a subset of the lean one.
            for r in &with {
                assert!(
                    without.contains(r),
                    "[{code}] {:?}: {r:?} matched with positions but not without",
                    case.query
                );
            }
            if case.expect_empty {
                assert!(
                    without.is_empty(),
                    "[{code}] {:?}: expected nothing, a positions-free index \
                     returned {without:?}",
                    case.query
                );
            }
        }
    }
}
