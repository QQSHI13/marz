//! Measure ranking precision on a real corpus.
//!
//! Usage: cargo run --release --example rankquality -- <corpus.json> <lang> <query>...
//!
//! Ground truth: a chunk is relevant when its source article's title contains
//! the query string. On a Wikipedia corpus this is a reasonable proxy — the
//! article titled 検索エンジン really is the one about search engines — and it is
//! independent of the scoring code being measured.
//!
//! Reports precision@k and mean average precision, so a change to scoring can
//! be judged by whether it moves those numbers rather than by reading the top
//! few hits and forming an impression.

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

    let mut sum_p1 = 0.0;
    let mut sum_p5 = 0.0;
    let mut sum_p10 = 0.0;
    let mut sum_ap = 0.0;
    let mut evaluated = 0;

    for query in &args[3..] {
        // Relevant = the chunk's location (article title + chunk index)
        // contains the query. Spaces are stripped so a Korean article titled
        // "검색 엔진" counts for the query "검색엔진".
        let needle: String = query.chars().filter(|c| !c.is_whitespace()).collect();
        let relevant = |loc: &str| -> bool {
            loc.chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>()
                .contains(&needle)
        };
        let total_relevant = docs
            .iter()
            .filter(|d| relevant(d["location"].as_str().unwrap()))
            .count();
        if total_relevant == 0 {
            println!("{query:16} SKIP (no relevant docs in corpus)");
            continue;
        }

        let results = index.search(query).unwrap();
        let hits: Vec<bool> = results.iter().map(|r| relevant(&r.ref_id)).collect();

        let precision_at = |k: usize| -> f64 {
            let k = k.min(hits.len());
            if k == 0 {
                return 0.0;
            }
            hits[..k].iter().filter(|h| **h).count() as f64 / k as f64
        };

        // Average precision: mean of precision@i over the ranks of relevant hits.
        let mut found = 0usize;
        let mut ap_sum = 0.0;
        for (i, hit) in hits.iter().enumerate() {
            if *hit {
                found += 1;
                ap_sum += found as f64 / (i + 1) as f64;
            }
        }
        let ap = if total_relevant > 0 {
            ap_sum / total_relevant as f64
        } else {
            0.0
        };

        println!(
            "{query:16} p@1 {:.2}  p@5 {:.2}  p@10 {:.2}  AP {:.3}   \
             ({} results, {} relevant)",
            precision_at(1),
            precision_at(5),
            precision_at(10),
            ap,
            results.len(),
            total_relevant,
        );

        sum_p1 += precision_at(1);
        sum_p5 += precision_at(5);
        sum_p10 += precision_at(10);
        sum_ap += ap;
        evaluated += 1;
    }

    if evaluated > 0 {
        let n = evaluated as f64;
        println!(
            "\nmean over {evaluated} queries: p@1 {:.3}  p@5 {:.3}  p@10 {:.3}  MAP {:.3}",
            sum_p1 / n,
            sum_p5 / n,
            sum_p10 / n,
            sum_ap / n
        );
    }
}
