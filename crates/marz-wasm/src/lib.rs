//! WebAssembly bindings for Marz.
//!
//! Provides a `MarzIndex` class that can load a serialized Marz index and
//! search it from JavaScript/Web Workers.

use std::sync::Arc;

use marz_core::languages::{Chinese, English, Japanese, Korean};
use marz_core::{Index, Language, SearchResult};
use serde::Serialize;
use wasm_bindgen::prelude::*;

fn language_from_code(code: &str) -> Option<Arc<dyn Language>> {
    match code {
        "en" => Some(Arc::new(English)),
        "zh" => Some(Arc::new(Chinese)),
        "ja" => Some(Arc::new(Japanese)),
        "ko" => Some(Arc::new(Korean)),
        _ => None,
    }
}

/// Serializable search result for JavaScript.
#[derive(Serialize)]
struct JsSearchResult {
    #[serde(rename = "ref")]
    ref_id: String,
    score: f64,
    terms: serde_json::Map<String, serde_json::Value>,
}

impl From<&SearchResult> for JsSearchResult {
    fn from(result: &SearchResult) -> Self {
        let mut terms = serde_json::Map::new();
        for (term, fields) in &result.match_data.terms {
            let mut field_map = serde_json::Map::new();
            for (field, positions) in fields {
                let positions: Vec<Vec<usize>> = positions
                    .iter()
                    .map(|(start, len)| vec![*start, *len])
                    .collect();
                field_map.insert(
                    field.clone(),
                    serde_json::to_value(positions).unwrap_or_default(),
                );
            }
            terms.insert(term.clone(), serde_json::Value::Object(field_map));
        }
        Self {
            ref_id: result.ref_id.clone(),
            score: result.score,
            terms,
        }
    }
}

/// A loaded Marz search index.
#[wasm_bindgen]
pub struct MarzIndex {
    index: Index,
}

#[wasm_bindgen]
impl MarzIndex {
    /// Load an index from a JSON string and a language code (`en`, `zh`, `ja`, `ko`).
    #[wasm_bindgen(constructor)]
    pub fn new(json: &str, lang_code: &str) -> Result<MarzIndex, JsValue> {
        let language = language_from_code(lang_code).ok_or_else(|| {
            JsValue::from_str(&format!("unsupported language code: {}", lang_code))
        })?;
        let index = Index::load(json, language).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(MarzIndex { index })
    }

    /// Search the index with a lunr query string.
    ///
    /// Returns an array of result objects: `{ ref, score, terms }`.
    pub fn search(&self, query: &str) -> Result<JsValue, JsValue> {
        let results = self
            .index
            .search(query)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let js_results: Vec<JsSearchResult> = results.iter().map(JsSearchResult::from).collect();
        serde_wasm_bindgen::to_value(&js_results).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Return the Marz core version.
    #[wasm_bindgen(getter)]
    pub fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}

/// Build a serialized index from a JSON documents array and language code.
///
/// Input format: `[{ "location": "...", "title": "...", "text": "..." }, ...]`.
#[wasm_bindgen]
pub fn build_index(docs_json: &str, lang_code: &str) -> Result<String, JsValue> {
    let language = language_from_code(lang_code)
        .ok_or_else(|| JsValue::from_str(&format!("unsupported language code: {}", lang_code)))?;

    let docs: Vec<serde_json::Map<String, serde_json::Value>> =
        serde_json::from_str(docs_json).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut builder = marz_core::IndexBuilder::new(language);
    builder
        .ref_field("location")
        .field("title", 1.0)
        .field("text", 1.0);

    for doc in &docs {
        let doc_ref = doc
            .get("location")
            .and_then(|v| v.as_str())
            .ok_or_else(|| JsValue::from_str("document missing location"))?;
        let boost = doc.get("boost").and_then(|v| v.as_f64()).unwrap_or(1.0);
        builder.add(doc_ref, boost, |name| {
            doc.get(name)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        });
    }

    Ok(builder.build().to_json())
}
