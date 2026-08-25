use std::sync::Arc;

use marz_core::languages::English;
use marz_core::{Index, IndexBuilder, Language};

fn build_test_index() -> (Index, String) {
    let language: Arc<dyn Language> = Arc::new(English);
    let mut builder = IndexBuilder::new(language.clone());
    builder
        .ref_field("id")
        .field("title", 1.0)
        .field("body", 1.0);
    builder.add("a", 1.0, |name| match name {
        "title" => Some("Mr. Green kills Colonel Mustard".to_string()),
        "body" => {
            Some("Mr. Green killed Colonel Mustard in the study with the candlestick.".to_string())
        }
        _ => None,
    });
    builder.add("b", 1.0, |name| match name {
        "title" => Some("Plumb water green plants".to_string()),
        "body" => Some("Professor Plumb has a green plant in his study".to_string()),
        _ => None,
    });
    let index = builder.build();
    let json = index.to_json();
    (index, json)
}

#[test]
fn serialization_includes_positions_and_stats() {
    let (_, json) = build_test_index();
    // Positions, for highlighting and phrase verification.
    assert!(json.contains("\"p\""));
    // Term frequencies and field lengths, which replace the precomputed
    // field vectors and are what makes query-time BM25 possible.
    assert!(json.contains("\"tf\""));
    assert!(json.contains("\"fieldLengths\""));
    assert!(json.contains("\"invertedIndex\""));
    // Field vectors must be gone, not merely unused.
    assert!(!json.contains("fieldVectors"));
}

#[test]
fn scores_survive_a_roundtrip_exactly() {
    // Scoring inputs are integers (term frequency, field length, document
    // count), so a roundtrip should reproduce scores bit-for-bit rather than
    // approximately.
    let (original, json) = build_test_index();
    let language: Arc<dyn Language> = Arc::new(English);
    let loaded = Index::load(&json, language).unwrap();

    for query in ["green", "study", "+green +candlestick", "pl*", "stud~1"] {
        let a = original.search(query).unwrap();
        let b = loaded.search(query).unwrap();
        assert_eq!(a.len(), b.len(), "result count differs for {query:?}");
        for (orig, load) in a.iter().zip(b.iter()) {
            assert_eq!(orig.ref_id, load.ref_id, "order differs for {query:?}");
            assert_eq!(orig.score, load.score, "score differs for {query:?}");
        }
    }
}

#[test]
fn loaded_index_returns_positions() {
    let (_, json) = build_test_index();
    let language: Arc<dyn Language> = Arc::new(English);
    let loaded = Index::load(&json, language).unwrap();

    let results = loaded.search("green").unwrap();
    let result = results
        .iter()
        .find(|r| r.ref_id == "b")
        .expect("doc b matched");

    let mut found_position = false;
    for fields in result.match_data.terms.values() {
        for positions in fields.values() {
            if !positions.is_empty() {
                found_position = true;
            }
        }
    }
    assert!(found_position, "expected non-empty positions in match data");
}
