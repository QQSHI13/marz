//! Report index size statistics and byte composition for a corpus.
//!
//! Usage: cargo run --release --example indexstats -- <corpus.json> <lang> [dump.json]
//!
//! Reports both serialization formats side by side, broken down by section, so a
//! change to either one can be attributed to the part that moved.

use std::collections::HashMap;
use std::sync::Arc;

use marz_core::languages::{Chinese, English, Japanese, Korean};
use marz_core::{BinaryIndex, IndexBuilder, Language};
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
            for docs in posting.values() {
                let Some(docs) = docs.as_object() else {
                    continue;
                };
                for meta in docs.values() {
                    postings += 1;
                    if let Some(Value::Array(p)) = meta.get("p") {
                        positions += p.len();
                    }
                }
            }
        }
        println!("\npostings        {postings}");
        println!("positions       {positions}");
    }

    // The binary format, section by section, against the JSON baseline.
    let t1 = std::time::Instant::now();
    let binary = idx.to_binary(true);
    let binary_ms = t1.elapsed().as_millis();
    let binary_no_pos = idx.to_binary(false);

    let view = BinaryIndex::open(&binary).expect("written index must open");
    let header = *view.header();
    let sections = [
        ("header", 0u32, header.meta_offset),
        ("meta", header.meta_offset, header.docs_offset),
        ("docs", header.docs_offset, header.terms_offset),
        ("terms", header.terms_offset, header.postings_offset),
        ("postings", header.postings_offset, header.positions_offset),
        ("positions", header.positions_offset, header.end_offset),
    ];

    println!("\nbinary format:");
    println!(
        "  total         {:10} bytes  {:.2}x smaller than json",
        binary.len(),
        json.len() as f64 / binary.len() as f64
    );
    println!(
        "  no positions  {:10} bytes  {:.2}x smaller than json",
        binary_no_pos.len(),
        json.len() as f64 / binary_no_pos.len() as f64
    );
    println!("  serialize     {binary_ms} ms");
    println!("\n  section breakdown:");
    for (name, start, end) in sections {
        let size = (end - start) as usize;
        println!(
            "    {name:12} {size:10}  {:5.1}%",
            100.0 * size as f64 / binary.len() as f64
        );
    }

    // Loading is what a search client pays on startup, so it is worth knowing
    // alongside the size.
    let t2 = std::time::Instant::now();
    let from_json = marz_core::Index::load(&json, lang.clone()).unwrap();
    let json_load_ms = t2.elapsed().as_millis();
    let t3 = std::time::Instant::now();
    let from_binary = marz_core::Index::from_binary(&binary, lang.clone()).unwrap();
    let binary_load_ms = t3.elapsed().as_millis();
    println!("\nload into memory:");
    println!("  from json     {json_load_ms} ms");
    println!("  from binary   {binary_load_ms} ms");
    assert_eq!(from_json.term_count(), from_binary.term_count());
    assert_eq!(from_json.document_count(), from_binary.document_count());

    if let Some(out) = args.get(3) {
        std::fs::write(out, &json).unwrap();
        std::fs::write(format!("{out}.marz"), &binary).unwrap();
        println!("\nwrote {out} and {out}.marz");
    }
}
