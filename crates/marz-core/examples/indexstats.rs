//! Report index size statistics and byte composition for a corpus.
//!
//! Usage: cargo run --release --example indexstats -- <corpus.json> <lang> [dump.json]
//!
//! This measures the baseline the binary index format has to beat, and breaks
//! the JSON down by section so the rewrite targets the sections that actually
//! cost bytes.

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

    let source_chars: usize = docs
        .iter()
        .map(|d| {
            ["title", "text"]
                .iter()
                .filter_map(|f| d.get(*f).and_then(|v| v.as_str()))
                .map(|s| s.chars().count())
                .sum::<usize>()
        })
        .sum();

    let mut token_count = 0usize;
    let mut unique_terms = std::collections::HashSet::new();
    for d in &docs {
        for f in ["title", "text"] {
            if let Some(s) = d.get(f).and_then(|v| v.as_str()) {
                for t in lang.tokenize(s) {
                    token_count += 1;
                    unique_terms.insert(t.term);
                }
            }
        }
    }

    let t0 = std::time::Instant::now();
    let mut b = IndexBuilder::new(lang.clone());
    b.ref_field("location")
        .field("title", 10.0)
        .field("text", 1.0);
    for d in &docs {
        b.add(d.get("location").unwrap().as_str().unwrap(), 1.0, |n| {
            d.get(n).and_then(|v| v.as_str()).map(|s| s.to_string())
        });
    }
    let idx = b.build();
    let build_ms = t0.elapsed().as_millis();

    let json = idx.to_json();

    println!("lang            {}", args[2]);
    println!("docs            {}", docs.len());
    println!("source chars    {source_chars}");
    println!(
        "tokens emitted  {token_count}  ({:.2} per source char)",
        token_count as f64 / source_chars as f64
    );
    println!("unique terms    {}", unique_terms.len());
    println!(
        "index json      {} bytes  ({:.1}x source chars)",
        json.len(),
        json.len() as f64 / source_chars as f64
    );
    println!("build           {build_ms} ms");

    // Break the serialized JSON down by top-level section.
    let parsed: Value = serde_json::from_str(&json).unwrap();
    println!("\nsection breakdown:");
    if let Value::Object(map) = &parsed {
        let mut rows: Vec<(String, usize)> = map
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::to_string(v).unwrap().len()))
            .collect();
        rows.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        for (k, n) in rows {
            println!(
                "  {k:16} {n:10}  {:5.1}%",
                100.0 * n as f64 / json.len() as f64
            );
        }
    }

    // Count postings and stored positions.
    if let Some(Value::Array(ii)) = parsed.get("invertedIndex") {
        let mut postings = 0usize;
        let mut positions = 0usize;
        for entry in ii {
            let Some(posting) = entry.get(1).and_then(|v| v.as_object()) else {
                continue;
            };
            for (field, docs) in posting {
                if field == "_index" {
                    continue;
                }
                let Some(docs) = docs.as_object() else {
                    continue;
                };
                for (_, meta) in docs {
                    postings += 1;
                    if let Some(Value::Array(p)) = meta.get("position") {
                        positions += p.len();
                    }
                }
            }
        }
        println!("\npostings        {postings}");
        println!("positions       {positions}");
    }
    if let Some(Value::Array(fv)) = parsed.get("fieldVectors") {
        let floats: usize = fv
            .iter()
            .filter_map(|e| e.get(1).and_then(|v| v.as_array()))
            .map(|a| a.len())
            .sum();
        println!("fieldVectors    {} entries, {} floats", fv.len(), floats);
    }

    if let Some(out) = args.get(3) {
        std::fs::write(out, &json).unwrap();
        println!("\nwrote {out}");
    }
}
