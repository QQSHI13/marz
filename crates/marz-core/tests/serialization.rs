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
fn serialization_includes_positions() {
    let (_, json) = build_test_index();
    assert!(json.contains("\"position\""));
    assert!(json.contains("\"_index\""));
    assert!(json.contains("\"fieldVectors\""));
    assert!(json.contains("\"invertedIndex\""));
}

#[test]
fn loaded_index_produces_same_results() {
    let (original, json) = build_test_index();
    let language: Arc<dyn Language> = Arc::new(English);
    let loaded = Index::load(&json, language).unwrap();

    let original_results = original.search("green").unwrap();
    let loaded_results = loaded.search("green").unwrap();

    assert_eq!(original_results.len(), loaded_results.len());
    for (orig, load) in original_results.iter().zip(loaded_results.iter()) {
        assert_eq!(orig.ref_id, load.ref_id);
        assert!((orig.score - load.score).abs() < 1e-9);
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
