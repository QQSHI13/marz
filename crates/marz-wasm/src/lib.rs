//! WebAssembly bindings for Marz.
//!
//! This is the *search* half of Marz. An index is built ahead of time — by a
//! static-site generator running the Python bindings, or by `marz-core` in a
//! build script — serialized with `to_binary()`, and shipped alongside the site.
//! This crate loads those bytes in a browser and queries them.
//!
//! # Why the surface is bytes and plain objects
//!
//! Two decisions here are about the size of what a visitor downloads, which for
//! a search client is the whole point.
//!
//! The load path takes a `Uint8Array` of the binary format and nothing else.
//! There is no JSON entry point: the JSON index is roughly five times the bytes
//! (2.54 MB against 439 KB on a Japanese corpus), and reading it would link
//! `serde_json`'s parser into a binary every visitor downloads in order to
//! support a format none of them should be served.
//!
//! Results are built as plain JavaScript objects through `js_sys::Reflect`
//! rather than derived with `serde-wasm-bindgen`. The two produce identical
//! objects; the derive cost 55 KB of WebAssembly — a quarter of this module — to
//! generate what the loop in [`MarzIndex::search`] writes in twenty lines.
//! Dropping it took the shipped binary from 229,720 bytes to 174,473
//! (106,918 to 83,734 gzipped).
//!
//! # Building an index here
//!
//! Off by default, behind the `builder` feature. Enabling it adds 39 KB for
//! tokenization, scoring and serialization, which is dead weight in a bundle
//! that only searches. Turn it on for the case that needs it — indexing content
//! that only exists client-side:
//!
//! ```sh
//! wasm-pack build crates/marz-wasm --features builder --target web
//! ```

use std::sync::Arc;

use marz_core::languages::{Chinese, English, Japanese, Korean};
use marz_core::{Index, Language};
use wasm_bindgen::prelude::*;

/// Language codes this build understands.
const LANGUAGE_CODES: [&str; 4] = ["en", "zh", "ja", "ko"];

/// Resolve a language code to its analysis rules.
///
/// A wrong code does not fail at load time — it produces an index whose query
/// tokenization disagrees with its indexed terms, so searches quietly return
/// nothing. That is why every entry point that names a language checks it here
/// instead of falling back to a default.
fn language_for(code: &str) -> Result<Arc<dyn Language>, JsValue> {
    match code {
        "en" => Ok(Arc::new(English)),
        "zh" => Ok(Arc::new(Chinese)),
        "ja" => Ok(Arc::new(Japanese)),
        "ko" => Ok(Arc::new(Korean)),
        other => Err(error(&format!(
            "unknown language code {other:?}; expected one of {}",
            LANGUAGE_CODES.join(", ")
        ))),
    }
}

/// Build a JavaScript `Error` to throw.
///
/// Every failure path goes through this rather than `JsValue::from_str`, which
/// throws a bare string. A thrown string has no `message` and no stack, so it
/// arrives in a caller's `catch` as something that cannot be logged usefully and
/// in a browser console as a line with no origin.
fn error(message: &str) -> JsValue {
    js_sys::Error::new(message).into()
}

/// Set a property, treating failure as impossible.
///
/// `Reflect::set` returns `Ok(false)` when a property is not writable. Every
/// object written here was created by `Object::new` one line earlier, so a
/// refusal cannot happen; propagating it would put a `?` on twenty lines to
/// describe a state that does not exist.
fn set(target: &js_sys::Object, key: &str, value: &JsValue) {
    let _ = js_sys::Reflect::set(target, &JsValue::from_str(key), value);
}

/// Language codes this build supports.
#[wasm_bindgen(js_name = "languages")]
pub fn languages() -> Vec<String> {
    LANGUAGE_CODES.iter().map(|c| (*c).to_string()).collect()
}

/// The version of Marz this module was built from.
#[wasm_bindgen(js_name = "version")]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Report the language an index was built for, without loading it.
///
/// Reads the 64-byte header and stops. Useful for picking which index to fetch
/// when a site ships one per language, and for asserting that a build pipeline
/// produced what it meant to.
#[wasm_bindgen(js_name = "indexLanguage")]
pub fn index_language(bytes: &[u8]) -> Result<String, JsValue> {
    marz_core::BinaryIndex::open(bytes)
        .map(|index| index.language().to_string())
        .map_err(|e| error(&format!("not a Marz index: {e}")))
}

/// Split `text` the way the index would, for inspecting how a query divides.
///
/// The fastest way to understand a CJK result: `tokenize("検索エンジン", "ja")`
/// returns `["検索", "エン", "ンジ", "ジン"]` — overlapping bigrams, except that
/// no bigram crosses the Han/Katakana boundary, because `索エ` spans two words
/// and would only add noise.
#[wasm_bindgen(js_name = "tokenize")]
pub fn tokenize(text: &str, language: &str) -> Result<Vec<String>, JsValue> {
    let language = language_for(language)?;
    Ok(language
        .tokenize(text)
        .into_iter()
        .map(|token| token.term)
        .collect())
}

/// Apply the same normalization the indexer applies before tokenizing.
///
/// Folds full-width Latin to ASCII, composes half-width katakana, collapses
/// exotic spaces and lowercases. Exported because match positions are offsets
/// into this string rather than into the input: normalization is not
/// length-preserving — `ｶﾞ` is two code points and becomes one `ガ` — so text
/// containing half-width katakana shifts every offset after it.
///
/// A caller highlighting a match must normalize the field text first. That is
/// what `highlight()` in the JavaScript wrapper does.
#[wasm_bindgen(js_name = "normalize")]
pub fn normalize(text: &str) -> String {
    marz_core::normalize::normalize(text)
}

/// TypeScript declarations for the plain objects [`MarzIndex::search`] returns.
///
/// These are hand-written because the objects are assembled through `Reflect`,
/// which wasm-bindgen cannot infer a shape from. They are what makes the
/// generated `.d.ts` describe a result instead of `any`.
#[wasm_bindgen(typescript_custom_section)]
const TYPESCRIPT_TYPES: &'static str = r#"
/**
 * Where a term matched, as `{ term: { field: [[start, length], ...] } }`.
 *
 * Offsets are measured in Unicode code points — not UTF-16 code units, and not
 * bytes — from the start of the *normalized* field text. Two consequences for a
 * caller that wants to highlight:
 *
 * 1. Normalize first. `normalize()` is exported for this. Normalization is not
 *    length-preserving, so offsets into the original string are wrong for any
 *    text containing half-width katakana.
 * 2. Slice by code point, not by `String.prototype.slice`, which counts UTF-16
 *    units and so drifts after any emoji or other astral-plane character.
 *    `[...text].slice(start, start + length).join("")` is correct.
 *
 * The `highlight()` helper in the JavaScript wrapper does both. An index built
 * without positions still reports which terms matched which fields, with empty
 * position arrays.
 */
export type Matches = Record<string, Record<string, Array<[number, number]>>>;

/** One search hit. */
export interface SearchResult {
  /** The matched document's reference, as passed to the builder. */
  ref: string;
  /** BM25 relevance score. Higher is better; comparable only within one result set. */
  score: number;
  /** Where each matched term occurred. For CJK the terms are bigrams, not words. */
  matches: Matches;
}
"#;

/// A loaded Marz search index.
///
/// Created with [`MarzIndex::load`]. Holds WebAssembly memory, so call `free()`
/// when done with it — or keep one for the lifetime of the page, which is the
/// normal shape for a search box.
#[wasm_bindgen]
pub struct MarzIndex {
    index: Index,
    language: String,
}

#[wasm_bindgen]
impl MarzIndex {
    /// Load an index from `to_binary()` output.
    ///
    /// The language is read from the index header, so it does not need to be
    /// passed. Give `expectedLanguage` to assert what the bytes should be: a
    /// build pipeline that ships the wrong file per locale otherwise produces a
    /// search box that finds nothing, with no error anywhere to explain it.
    ///
    /// This materializes the postings rather than reading them in place, so it
    /// costs time and memory proportional to the index — a 439 KB Japanese index
    /// takes roughly 20 ms. Do it once and keep the result.
    #[wasm_bindgen(js_name = "load")]
    pub fn load(bytes: &[u8], expected_language: Option<String>) -> Result<MarzIndex, JsValue> {
        // Read the header before the body, so that bytes which are not an index
        // at all report that, and a language mismatch is reported without first
        // paying for a load that is about to be thrown away.
        let stored = marz_core::BinaryIndex::open(bytes)
            .map_err(|e| error(&format!("not a Marz index: {e}")))?
            .language()
            .to_string();

        if let Some(expected) = expected_language {
            if expected != stored {
                return Err(error(&format!(
                    "index was built for language {stored:?}, not {expected:?}"
                )));
            }
        }

        let language = language_for(&stored)?;
        let index = Index::from_binary(bytes, language)
            .map_err(|e| error(&format!("could not read index: {e}")))?;

        Ok(MarzIndex {
            index,
            language: stored,
        })
    }

    /// Search the index, returning hits in descending score order.
    ///
    /// Query syntax: bare terms, `+required`, `-prohibited`, `field:term`,
    /// `term*` wildcards, `term~N` fuzzy matching and `term^N` boosts.
    ///
    /// `limit` caps how many hits are converted to JavaScript objects. Scoring
    /// happens for the whole corpus either way — the cap saves building position
    /// maps for results past the first page, which is where the conversion cost
    /// is.
    ///
    /// Throws an `Error` if the query cannot be parsed. The error carries
    /// `query` plus `start` and `end` offsets into it, enough to underline the
    /// fault in a search box.
    #[wasm_bindgen(unchecked_return_type = "SearchResult[]")]
    pub fn search(&self, query: &str, limit: Option<usize>) -> Result<js_sys::Array, JsValue> {
        let results = self.index.search(query).map_err(|e| {
            let err = js_sys::Error::new(&format!("{} in query {query:?}", e.message));
            // On the error rather than in the message, so that `err.message`
            // stays a sentence a user can be shown while a caller that wants to
            // highlight the span can still find it.
            let _ = js_sys::Reflect::set(&err, &"query".into(), &query.into());
            let _ = js_sys::Reflect::set(&err, &"start".into(), &(e.start as f64).into());
            let _ = js_sys::Reflect::set(&err, &"end".into(), &(e.end as f64).into());
            JsValue::from(err)
        })?;

        let take = limit.unwrap_or(usize::MAX);
        let out = js_sys::Array::new();
        for result in results.iter().take(take) {
            let hit = js_sys::Object::new();
            set(&hit, "ref", &JsValue::from_str(&result.ref_id));
            set(&hit, "score", &JsValue::from_f64(result.score));

            let matches = js_sys::Object::new();
            for (term, fields) in &result.match_data.terms {
                let per_field = js_sys::Object::new();
                for (field, positions) in fields {
                    let spans = js_sys::Array::new();
                    for (start, length) in positions {
                        let span = js_sys::Array::new();
                        span.push(&JsValue::from_f64(*start as f64));
                        span.push(&JsValue::from_f64(*length as f64));
                        spans.push(&span);
                    }
                    set(&per_field, field, &spans);
                }
                set(&matches, term, &per_field);
            }
            set(&hit, "matches", &matches);
            out.push(&hit);
        }
        Ok(out)
    }

    /// The language code this index was built for.
    #[wasm_bindgen(getter)]
    pub fn language(&self) -> String {
        self.language.clone()
    }

    /// The indexed field names.
    #[wasm_bindgen(getter)]
    pub fn fields(&self) -> Vec<String> {
        self.index.fields().to_vec()
    }

    /// The number of indexed documents.
    #[wasm_bindgen(getter, js_name = "documentCount")]
    pub fn document_count(&self) -> usize {
        self.index.document_count()
    }

    /// The number of distinct indexed terms. For CJK, bigrams rather than words.
    #[wasm_bindgen(getter, js_name = "termCount")]
    pub fn term_count(&self) -> usize {
        self.index.term_count()
    }
}

impl MarzIndex {
    /// Wrap an already-built core index. Not exposed to JavaScript.
    ///
    /// Lets the optional builder hand over an index directly instead of
    /// serializing to bytes and immediately reading them back.
    #[cfg(feature = "builder")]
    pub(crate) fn from_parts(index: Index, language: String) -> Self {
        MarzIndex { index, language }
    }
}

#[cfg(feature = "builder")]
mod builder;
