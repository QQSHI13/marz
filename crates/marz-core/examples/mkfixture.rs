//! Write a binary index from a JSON corpus, for the JS test harness.
use marz_core::languages::registry;
use marz_core::{IndexBuilder, Language};
use std::sync::Arc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (input, output, lang) = (&args[1], &args[2], &args[3]);
    let raw = std::fs::read_to_string(input).unwrap();
    let docs: Vec<serde_json::Map<String, serde_json::Value>> = serde_json::from_str(&raw).unwrap();
    // Fail loudly on a typoed language: resolving silently falls back to
    // generic tokenization, and the baked fixture would then pin the wrong
    // behavior into every JS test run.
    let resolved = registry::resolve(lang);
    if !resolved.exact {
        eprintln!("unknown language code {lang:?}: refusing to bake a fixture");
        std::process::exit(1);
    }
    let language: Arc<dyn Language> = resolved.language;
    let mut b = IndexBuilder::new(language);
    b.ref_field("location")
        .field("title", 10.0)
        .field("text", 1.0);
    for doc in &docs {
        let r = doc["location"].as_str().unwrap();
        b.add(r, 1.0, |n| {
            doc.get(n).and_then(|v| v.as_str()).map(str::to_string)
        });
    }
    let idx = b.build();
    let bytes = idx.to_binary(true);
    std::fs::write(output, &bytes).unwrap();
    eprintln!(
        "{} docs, {} terms, {} bytes",
        idx.document_count(),
        idx.term_count(),
        bytes.len()
    );
}
