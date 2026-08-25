//! Smoke-test search quality on a real corpus.
//!
//! Usage: cargo run --release --example searchprobe -- <corpus.json> <lang> <query>...

use std::collections::HashMap;
use std::sync::Arc;

use marz_core::languages::{Chinese, English, Japanese, Korean};
use marz_core::{IndexBuilder, Language};
use serde_json::Value;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let docs: Vec<HashMap<String, Value>> =
        serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let lang: Arc<dyn Language> = match args[2].as_str() {
        "zh" => Arc::new(Chinese),
        "ja" => Arc::new(Japanese),
        "ko" => Arc::new(Korean),
        _ => Arc::new(English),
    };

    let mut b = IndexBuilder::new(lang.clone());
    b.ref_field("location")
        .field("title", 10.0)
        .field("text", 1.0);
    for d in &docs {
        b.add(d.get("location").unwrap().as_str().unwrap(), 1.0, |n| {
            d.get(n).and_then(|v| v.as_str()).map(|s| s.to_string())
        });
    }
    let index = b.build();
    println!(
        "{} docs, {} terms\n",
        index.document_count(),
        index.term_count()
    );

    for query in &args[3..] {
        let tokens: Vec<String> = lang.tokenize(query).into_iter().map(|t| t.term).collect();
        let results = index.search(query).unwrap();
        println!("query {query:?}  -> tokens {tokens:?}");
        if results.is_empty() {
            println!("  NO RESULTS");
        }
        for r in results.iter().take(5) {
            println!("  {:8.3}  {}", r.score, r.ref_id);
        }
        println!("  ({} total)\n", results.len());
    }
}
