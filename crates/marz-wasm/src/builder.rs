//! Index building in the browser, behind the `builder` feature.
//!
//! The normal shape for Marz is to build ahead of time and ship the bytes, which
//! is why this is off by default: it adds 39 KB of WebAssembly (174,473 bytes to
//! 213,044) for tokenization, scoring and serialization that a search box never
//! executes.
//!
//! It exists for content that only exists client-side — a note-taking app whose
//! documents live in IndexedDB, or a viewer indexing a file the user just
//! dropped on the page. There is no build step to hook into there.
//!
//! Documents come in as JavaScript objects read through `Reflect`, not as a JSON
//! string. A caller in a browser already has objects; making them
//! `JSON.stringify` first so this crate can link a JSON parser to undo it would
//! cost both sides.

use std::collections::HashMap;

use marz_core::IndexBuilder as CoreBuilder;
use wasm_bindgen::prelude::*;

use crate::{error, language_for};

/// A document staged for indexing, already copied out of JavaScript.
struct StagedDoc {
    doc_ref: String,
    boost: f64,
    fields: HashMap<String, String>,
}

/// Read a field out of a JavaScript object.
///
/// Absent, `undefined` and `null` all mean "this document has no such field",
/// which is ordinary — an optional summary, say. A field holding a non-string is
/// a bug in the caller's data and throws rather than being coerced: letting a
/// number through would index `"1"` as searchable text, and letting an object
/// through would index `"[object Object]"`, neither of which surfaces until a
/// search comes back wrong.
fn field_text(doc: &JsValue, name: &str) -> Result<Option<String>, JsValue> {
    let value = js_sys::Reflect::get(doc, &JsValue::from_str(name))?;
    if value.is_undefined() || value.is_null() {
        return Ok(None);
    }
    match value.as_string() {
        Some(text) => Ok(Some(text)),
        None => Err(error(&format!(
            "field {name:?} must be a string, null or undefined"
        ))),
    }
}

/// Builds a search index in the browser.
///
/// ```js
/// const b = new MarzBuilder("ja", "location");
/// b.field("title", 10.0);
/// b.field("text");
/// b.add({ location: "guide/intro", title: "入門", text: "…" });
/// const bytes = b.build();          // Uint8Array, ready to store or search
/// ```
#[wasm_bindgen]
pub struct MarzBuilder {
    language_code: String,
    ref_field: String,
    fields: Vec<(String, f64)>,
    docs: Vec<StagedDoc>,
    k1: f64,
    b: f64,
}

#[wasm_bindgen]
impl MarzBuilder {
    /// Create a builder for `language`, one of the codes `languages()` returns.
    ///
    /// `refField` names the property holding each document's identity and
    /// defaults to `"id"`. `k1` and `b` are the BM25 tuning parameters; the
    /// defaults are the conventional 1.2 and 0.75.
    #[wasm_bindgen(constructor)]
    pub fn new(
        language: &str,
        ref_field: Option<String>,
        k1: Option<f64>,
        b: Option<f64>,
    ) -> Result<MarzBuilder, JsValue> {
        // Resolved and discarded: this is here to reject a bad code at
        // construction. Deferring it to `build` would let a caller stage a
        // thousand documents before learning the language was misspelled.
        language_for(language)?;
        Ok(MarzBuilder {
            language_code: language.to_string(),
            ref_field: ref_field.unwrap_or_else(|| "id".to_string()),
            fields: Vec::new(),
            docs: Vec::new(),
            k1: k1.unwrap_or(1.2),
            b: b.unwrap_or(0.75),
        })
    }

    /// Declare a searchable field. `boost` multiplies the score of matches in it.
    ///
    /// Fields must be declared before the documents that use them: `add` reads
    /// only the fields declared when it is called.
    pub fn field(&mut self, name: &str, boost: Option<f64>) -> Result<(), JsValue> {
        if self.fields.iter().any(|(existing, _)| existing == name) {
            return Err(error(&format!("field {name:?} is already declared")));
        }
        if name == self.ref_field {
            // Legal, but almost always a mistake: it makes every document match
            // its own identifier, which inflates scores in a way that is hard to
            // trace back to this line.
            return Err(error(&format!(
                "field {name:?} is the reference field; \
                 pass a different refField to index it as text"
            )));
        }
        self.fields.push((name.to_string(), boost.unwrap_or(1.0)));
        Ok(())
    }

    /// Stage a document for indexing.
    ///
    /// The reference field must be present and a string; searchable fields may
    /// be absent, `null` or `undefined`.
    pub fn add(&mut self, doc: &JsValue, boost: Option<f64>) -> Result<(), JsValue> {
        if self.fields.is_empty() {
            return Err(error(
                "declare at least one field with field() before adding documents",
            ));
        }
        let doc_ref = field_text(doc, &self.ref_field)?.ok_or_else(|| {
            error(&format!(
                "document is missing its reference field {:?}",
                self.ref_field
            ))
        })?;

        let mut fields = HashMap::with_capacity(self.fields.len());
        for (name, _) in &self.fields {
            if let Some(text) = field_text(doc, name)? {
                fields.insert(name.clone(), text);
            }
        }
        self.docs.push(StagedDoc {
            doc_ref,
            boost: boost.unwrap_or(1.0),
            fields,
        });
        Ok(())
    }

    /// Stage every document in an iterable. Equivalent to `add` in a loop.
    ///
    /// Not atomic, for the same reason: if a later document throws, the earlier
    /// ones stay staged. Call `clear()` if a half-filled builder is not wanted.
    #[wasm_bindgen(js_name = "addMany")]
    pub fn add_many(&mut self, docs: &JsValue, boost: Option<f64>) -> Result<(), JsValue> {
        let iterator = js_sys::try_iter(docs)?
            .ok_or_else(|| error("addMany expects an array or other iterable"))?;
        for doc in iterator {
            self.add(&doc?, boost)?;
        }
        Ok(())
    }

    /// Tokenize and score the staged documents, returning the binary index.
    ///
    /// Pass `positions = false` to drop highlighting and CJK phrase
    /// verification, which is about a tenth of the bytes.
    ///
    /// The staged documents are kept, so building twice yields two equivalent
    /// indexes rather than an index and an empty one. Building is not cheap, but
    /// a `build()` that quietly emptied the builder would turn a stray second
    /// call into a silently empty search index.
    pub fn build(&self, positions: Option<bool>) -> Result<Vec<u8>, JsValue> {
        let language = language_for(&self.language_code)?;
        let mut builder = CoreBuilder::new(language);
        builder
            .ref_field(self.ref_field.clone())
            .k1(self.k1)
            .b(self.b);
        for (name, boost) in &self.fields {
            builder.field(name.clone(), *boost);
        }
        for doc in &self.docs {
            builder.add(doc.doc_ref.clone(), doc.boost, |name| {
                doc.fields.get(name).cloned()
            });
        }
        Ok(builder.build().to_binary(positions.unwrap_or(true)))
    }

    /// Build and load in one step, skipping the serialize/parse round trip.
    ///
    /// For the client-side case this is the common path: the index is being
    /// built to be searched now, not stored. Use `build()` when the bytes are
    /// what is wanted, to cache in IndexedDB or hand to a worker.
    #[wasm_bindgen(js_name = "buildAndLoad")]
    pub fn build_and_load(&self) -> Result<crate::MarzIndex, JsValue> {
        let language = language_for(&self.language_code)?;
        let mut builder = CoreBuilder::new(language);
        builder
            .ref_field(self.ref_field.clone())
            .k1(self.k1)
            .b(self.b);
        for (name, boost) in &self.fields {
            builder.field(name.clone(), *boost);
        }
        for doc in &self.docs {
            builder.add(doc.doc_ref.clone(), doc.boost, |name| {
                doc.fields.get(name).cloned()
            });
        }
        Ok(crate::MarzIndex::from_parts(
            builder.build(),
            self.language_code.clone(),
        ))
    }

    /// Discard the staged documents, keeping the field configuration.
    pub fn clear(&mut self) {
        self.docs.clear();
    }

    /// How many documents are staged and ready to build.
    #[wasm_bindgen(getter)]
    pub fn staged(&self) -> usize {
        self.docs.len()
    }

    /// The declared field names, in declaration order.
    #[wasm_bindgen(getter)]
    pub fn fields(&self) -> Vec<String> {
        self.fields.iter().map(|(name, _)| name.clone()).collect()
    }

    /// The configured reference field.
    #[wasm_bindgen(getter, js_name = "refField")]
    pub fn ref_field(&self) -> String {
        self.ref_field.clone()
    }

    /// The configured language code.
    #[wasm_bindgen(getter)]
    pub fn language(&self) -> String {
        self.language_code.clone()
    }
}
