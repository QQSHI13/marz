use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use marz_core::languages::English;
use marz_core::{IndexBuilder, Language};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
struct ExpectedResult {
    #[serde(rename = "ref")]
    ref_id: String,
    score: f64,
}

#[test]
fn golden_english_parity() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");

    let documents_file = File::open(fixtures.join("documents.json")).unwrap();
    let documents: Vec<HashMap<String, Value>> = serde_json::from_reader(documents_file).unwrap();

    let queries_file = File::open(fixtures.join("queries.json")).unwrap();
    let expected: HashMap<String, Vec<ExpectedResult>> =
        serde_json::from_reader(queries_file).unwrap();

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

    let index = builder.build();

    for (query, expected_results) in expected {
        let results = index.search(&query).unwrap();
        assert_eq!(
            results.len(),
            expected_results.len(),
            "query '{}' result count mismatch",
            query
        );
        for (i, expected) in expected_results.iter().enumerate() {
            assert_eq!(results[i].ref_id, expected.ref_id, "query '{}'", query);
            let diff = (results[i].score - expected.score).abs();
            assert!(
                diff < 1e-5,
                "query '{}' ref '{}' score mismatch: expected {} got {} (diff {})",
                query,
                expected.ref_id,
                expected.score,
                results[i].score,
                diff
            );
        }
    }
}
