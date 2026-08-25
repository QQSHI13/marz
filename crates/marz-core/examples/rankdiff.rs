//! Compare Marz ranking against the lunr golden fixture.
//!
//! Usage: cargo run --release --example rankdiff

use std::collections::HashMap;
use std::sync::Arc;

use marz_core::languages::English;
use marz_core::{IndexBuilder, Language};
use serde_json::Value;

fn main() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    let documents: Vec<HashMap<String, Value>> =
        serde_json::from_str(&std::fs::read_to_string(fixtures.join("documents.json")).unwrap())
            .unwrap();
    let expected: HashMap<String, Vec<HashMap<String, Value>>> =
        serde_json::from_str(&std::fs::read_to_string(fixtures.join("queries.json")).unwrap())
            .unwrap();

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

    let mut queries: Vec<_> = expected.keys().cloned().collect();
    queries.sort();

    let mut order_ok = 0;
    let mut order_bad = 0;
    let mut set_bad = 0;

    for query in &queries {
        let want: Vec<String> = expected[query]
            .iter()
            .map(|r| r["ref"].as_str().unwrap().to_string())
            .collect();
        let got: Vec<String> = index
            .search(query)
            .unwrap()
            .into_iter()
            .map(|r| r.ref_id)
            .collect();

        let mut ws = want.clone();
        let mut gs = got.clone();
        ws.sort();
        gs.sort();

        if ws != gs {
            set_bad += 1;
            println!("SET  {query:24} want {want:?}\n     {:24} got  {got:?}", "");
        } else if want != got {
            order_bad += 1;
            println!("ORD  {query:24} want {want:?}\n     {:24} got  {got:?}", "");
        } else {
            order_ok += 1;
        }
    }

    println!("\n{} queries", queries.len());
    println!("  identical order   {order_ok}");
    println!("  same set, reorder {order_bad}");
    println!("  different set     {set_bad}");
}
