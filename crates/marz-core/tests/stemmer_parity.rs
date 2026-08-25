//! Stemmer parity against lunr.js.
//!
//! The repo-root fixture `tests/fixtures/stemmer_lunr_js.json` maps 9,953
//! English words to the stem produced by lunr.js 2.3.9, generated with
//! `tests/generate_stemmer_fixture.js`. The word list is a broad sample of
//! English vocabulary chosen to exercise every branch of the algorithm.
//!
//! lunr.js is the oracle, not lunr.py: the two disagree on terminal `y`
//! handling and cannot both be satisfied. See `languages::porter` for the
//! details.

use std::collections::BTreeMap;

use marz_core::languages::porter;

#[test]
fn stemmer_matches_lunr_js_on_full_word_list() {
    let raw = include_str!("../../../tests/fixtures/stemmer_lunr_js.json");
    let oracle: BTreeMap<String, String> =
        serde_json::from_str(raw).expect("parse stemmer fixture");
    assert!(
        oracle.len() > 9000,
        "fixture looks truncated: {} entries",
        oracle.len()
    );

    let mut mismatches = Vec::new();
    for (word, expected) in &oracle {
        let got = porter::stem(word);
        if &got != expected {
            mismatches.push(format!("{word}: lunr.js={expected} marz={got}"));
        }
    }

    assert!(
        mismatches.is_empty(),
        "{} of {} words disagree with lunr.js:\n  {}",
        mismatches.len(),
        oracle.len(),
        mismatches
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}
