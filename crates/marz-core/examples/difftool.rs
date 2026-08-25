use marz_core::languages::English;
use marz_core::{IndexBuilder, Language};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let docs: Vec<HashMap<String, Value>> =
        serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let queries: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(&args[2]).unwrap()).unwrap();

    let lang: Arc<dyn Language> = Arc::new(English);
    let mut b = IndexBuilder::new(lang);
    b.ref_field("location")
        .field("title", 10.0)
        .field("text", 1.0);
    for d in &docs {
        let boost = d.get("_boost").and_then(|v| v.as_f64()).unwrap_or(1.0);
        b.add(d.get("location").unwrap().as_str().unwrap(), boost, |n| {
            d.get(n).and_then(|v| v.as_str()).map(|s| s.to_string())
        });
    }
    let idx = b.build();

    let mut out = serde_json::Map::new();
    for q in &queries {
        match idx.search(q) {
            Ok(rs) => {
                let arr: Vec<Value> = rs.iter().map(|r| {
                    serde_json::json!({"ref": r.ref_id, "score": (r.score*1e6).round()/1e6})
                }).collect();
                out.insert(q.clone(), Value::Array(arr));
            }
            Err(e) => {
                out.insert(q.clone(), Value::String(format!("ERROR: {}", e)));
            }
        }
    }
    println!("{}", serde_json::to_string(&out).unwrap());
}
