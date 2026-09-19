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

use crate::{error, language_for, language_for_load};

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
    rehydrated: Option<CoreBuilder>,
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
        // Resolved and kept: the canonical code (`"en, ja"` → `"en,ja"`)
        // matches what the index header stores, so identity survives a
        // roundtrip. This also rejects nothing — unknown codes fall back with
        // a warning — but empty codes are always a bug. Resolving here rather
        // than at `build` surfaces a misspelling before a thousand documents
        // are staged.
        if language.trim().is_empty() {
            return Err(error("language must not be empty"));
        }
        let language_code = language_for(language).code().to_string();
        let ref_field = ref_field.unwrap_or_else(|| "id".to_string());
        if ref_field.trim().is_empty() {
            return Err(error("refField must not be empty"));
        }
        let ref_field = ref_field.trim().to_string();
        let k1 = match k1 {
            Some(v) if v.is_finite() && v >= 0.0 => v,
            // Clamp philosophy, matching the core and Python bindings: a
            // negative k1 keeps the default rather than raising. Only
            // non-finite values are programmer errors worth throwing for.
            Some(v) if v.is_finite() => 1.2,
            Some(_) => return Err(error("k1 must be a finite number")),
            None => 1.2,
        };
        let b = match b {
            Some(v) if v.is_finite() => v.clamp(0.0, 1.0),
            Some(_) => return Err(error("b must be a finite number")),
            None => 0.75,
        };
        Ok(MarzBuilder {
            language_code,
            ref_field,
            fields: Vec::new(),
            docs: Vec::new(),
            k1,
            b,
            rehydrated: None,
        })
    }

    /// Rehydrate a builder from a loaded index for incremental updates.
    ///
    /// Clones the index's postings into a builder, restoring the declared
    /// fields (names and boosts) and the language, so documents can be added
    /// or removed without re-adding the whole corpus. The reference field
    /// name is not stored in the index, so pass it again here when it is not
    /// `"id"`. `k1`/`b` come along with the index: this takes none, and the
    /// stored tuning is kept — there is nothing to set.
    ///
    /// Staged additions still go through `add` and are applied on `build`,
    /// which clones the restored state — building twice gives two equivalent
    /// indexes, as with a fresh builder.
    #[wasm_bindgen(js_name = "fromIndex")]
    pub fn from_index(
        index: &crate::MarzIndex,
        ref_field: Option<String>,
    ) -> Result<MarzBuilder, JsValue> {
        let ref_field = ref_field.unwrap_or_else(|| "id".to_string());
        if ref_field.trim().is_empty() {
            return Err(error("refField must not be empty"));
        }
        let (core_index, language_code) = index.clone_parts();
        let mut core = CoreBuilder::from_index(core_index);
        core.ref_field(ref_field.trim().to_string());
        let fields = core.declared_fields();
        Ok(MarzBuilder {
            language_code,
            ref_field: ref_field.trim().to_string(),
            fields,
            docs: Vec::new(),
            // Unused in rehydrated builds: the restored core builder already
            // carries the index's tuning.
            k1: 1.2,
            b: 0.75,
            rehydrated: Some(core),
        })
    }

    /// Declare a searchable field. `boost` multiplies the score of matches in it.
    ///
    /// Fields must be declared before the documents that use them: `add` reads
    /// only the fields declared when it is called. A negative `boost` is
    /// clamped to `0.0` at declaration (it still matches, contributing no
    /// score); only a non-finite boost throws.
    ///
    /// Surrounding whitespace is not part of a name: it is trimmed before
    /// storing, so `field(" title ")` and `title:q` meet.
    pub fn field(&mut self, name: &str, boost: Option<f64>) -> Result<(), JsValue> {
        let name = name.trim();
        if name.is_empty() {
            return Err(error("field name must not be empty"));
        }
        if name.contains('/') {
            return Err(error(&format!(
                "field {name:?} must not contain '/' (breaks FieldRef round-trip)"
            )));
        }
        // The builder only stores the code; resolve for the separator set.
        // Multi-language codes resolve to the union (same rule as search),
        // so validation agrees with indexing. (Core asserts the same
        // predicate as a backstop.)
        let separators = marz_core::languages::registry::resolve_multi(&self.language_code)
            .language
            .separator_chars()
            .to_string();
        if name
            .chars()
            .any(|c| c.is_whitespace() || separators.contains(c))
        {
            return Err(error(&format!(
                "field {name:?} contains a separator and is unqueryable via field: syntax"
            )));
        }
        if name.chars().any(|c| matches!(c, ':' | '^' | '~' | '\\'))
            || matches!(name.chars().next(), Some('+' | '-'))
        {
            return Err(error(&format!(
                "field {name:?} contains a query operator and is unqueryable \
                 via field:term syntax"
            )));
        }
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
        let boost = match boost {
            Some(v) if v.is_finite() => v.max(0.0),
            Some(_) => return Err(error("field boost must be a finite number")),
            None => 1.0,
        };
        self.fields.push((name.to_string(), boost));
        // A rehydrated builder indexes through the restored core builder, not
        // through a fresh one assembled at `build`, so a newly declared field
        // must reach both lists or its staged text would be silently dropped.
        if let Some(core) = self.rehydrated.as_mut() {
            core.field(name.to_string(), boost);
        }
        Ok(())
    }

    /// Stage a document for indexing.
    ///
    /// The reference field must be present and a string; searchable fields may
    /// be absent, `null` or `undefined`. Adding the same reference twice
    /// replaces the previous document (upsert). A negative `boost` is clamped
    /// to `0.0` at declaration; only a non-finite boost throws.
    pub fn add(&mut self, doc: &JsValue, boost: Option<f64>) -> Result<(), JsValue> {
        if !doc.is_object() {
            return Err(error("document must be an object"));
        }
        if self.fields.is_empty() {
            return Err(error(
                "declare at least one field with field() before adding documents",
            ));
        }
        let boost = match boost {
            Some(v) if v.is_finite() => v.max(0.0),
            Some(_) => return Err(error("document boost must be a finite number")),
            None => 1.0,
        };
        let doc_ref = field_text(doc, &self.ref_field)?.ok_or_else(|| {
            error(&format!(
                "document is missing its reference field {:?}",
                self.ref_field
            ))
        })?;
        if doc_ref.is_empty() {
            return Err(error("document reference must not be empty"));
        }

        let mut fields = HashMap::with_capacity(self.fields.len());
        for (name, _) in &self.fields {
            if let Some(text) = field_text(doc, name)? {
                fields.insert(name.clone(), text);
            }
        }
        let staged = StagedDoc {
            doc_ref,
            boost,
            fields,
        };
        // Upsert: same reference replaces, matching core `IndexBuilder::add`.
        if let Some(existing) = self.docs.iter_mut().find(|d| d.doc_ref == staged.doc_ref) {
            *existing = staged;
        } else {
            self.docs.push(staged);
        }
        Ok(())
    }

    /// Stage every document in an iterable. Equivalent to `add` in a loop.
    ///
    /// Not atomic, for the same reason: if a later document throws, the earlier
    /// ones stay staged. Call `clear()` if a half-filled builder is not wanted.
    #[wasm_bindgen(js_name = "addMany")]
    pub fn add_many(&mut self, docs: &JsValue, boost: Option<f64>) -> Result<(), JsValue> {
        let iterator = js_sys::try_iter(docs)
            .map_err(|e| {
                error(&format!(
                    "addMany expects an array or other iterable: {}",
                    js_sys::JSON::stringify(&e)
                        .ok()
                        .and_then(|s| s.as_string())
                        .unwrap_or_else(|| "unreadable error".to_string())
                ))
            })?
            .ok_or_else(|| error("addMany expects an array or other iterable"))?;
        for doc in iterator {
            let doc = doc.map_err(|e| {
                error(&format!(
                    "addMany failed to read a document: {}",
                    js_sys::JSON::stringify(&e)
                        .ok()
                        .and_then(|s| s.as_string())
                        .unwrap_or_else(|| "unreadable error".to_string())
                ))
            })?;
            self.add(&doc, boost)?;
        }
        Ok(())
    }

    /// Remove the document with reference `docRef`.
    ///
    /// Returns `true` when a document was present and removed, `false` when
    /// no document used that reference. Only a builder from `fromIndex` can
    /// remove: a fresh builder holds nothing to remove from, so it throws
    /// rather than silently returning `false`. Staged additions with the same
    /// reference are dropped as well, so a removed document stays removed at
    /// the next `build`. A removal already applied is not undone by `clear`.
    pub fn remove(&mut self, doc_ref: &str) -> Result<bool, JsValue> {
        let Some(core) = self.rehydrated.as_mut() else {
            return Err(error(
                "remove() needs a builder from MarzBuilder.fromIndex(); \
                 a fresh builder holds nothing to remove",
            ));
        };
        let staged = self.docs.iter().any(|d| d.doc_ref == doc_ref);
        self.docs.retain(|d| d.doc_ref != doc_ref);
        Ok(core.remove(doc_ref) || staged)
    }

    /// Shared core-builder construction (fields + docs + BM25 params).
    ///
    /// A rehydrated builder clones its restored state and applies the staged
    /// documents on top, so removals and upserts both survive repeated builds.
    fn core_builder(&self) -> CoreBuilder {
        if let Some(core) = &self.rehydrated {
            let mut builder = core.clone();
            for doc in &self.docs {
                builder.add(doc.doc_ref.clone(), doc.boost, |name| {
                    doc.fields.get(name).cloned()
                });
            }
            return builder;
        }
        // Silent resolve: `new` already warned for an unknown code; warning
        // again on every `build`/`buildAndLoad` would double-report one typo.
        let language = language_for_load(&self.language_code);
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
        builder
    }

    /// Tokenize and score the staged documents, returning the binary index.
    ///
    /// Pass `positions = false` to drop highlighting and CJK phrase
    /// verification, which is about a tenth of the bytes. This also disables
    /// the CJK phrase ranking boost.
    ///
    /// The staged documents are kept, so building twice yields two equivalent
    /// indexes rather than an index and an empty one. Building is not cheap, but
    /// a `build()` that quietly emptied the builder would turn a stray second
    /// call into a silently empty search index.
    pub fn build(&self, positions: Option<bool>) -> Result<Vec<u8>, JsValue> {
        self.core_builder()
            .build()
            .to_binary(positions.unwrap_or(true))
            .map_err(|e| error(&e.to_string()))
    }

    /// Build and load in one step, skipping the serialize/parse round trip.
    ///
    /// For the client-side case this is the common path: the index is being
    /// built to be searched now, not stored. Use `build()` when the bytes are
    /// what is wanted, to cache in IndexedDB or hand to a worker.
    ///
    /// Unlike `build()`, this takes no `positions` flag: the in-memory index
    /// always keeps positions (dropping them only shrinks the serialized
    /// bytes), so there is nothing to opt out of here.
    #[wasm_bindgen(js_name = "buildAndLoad")]
    pub fn build_and_load(&self) -> Result<crate::MarzIndex, JsValue> {
        Ok(crate::MarzIndex::from_parts(
            self.core_builder().build(),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-document fixture, built through core: `add` needs live JavaScript
    /// objects, which the host test runner does not have.
    fn fixture_bytes() -> Vec<u8> {
        let language = marz_core::languages::registry::resolve("en").language;
        let mut builder = CoreBuilder::new(language);
        builder
            .ref_field("id")
            .field("title", 10.0)
            .field("body", 1.0);
        builder.add("a", 1.0, |name| match name {
            "title" => Some("green apples".to_string()),
            "body" => Some("green orchard".to_string()),
            _ => None,
        });
        builder.add("b", 1.0, |name| match name {
            "title" => Some("plumb pipes".to_string()),
            "body" => Some("lead pipes".to_string()),
            _ => None,
        });
        builder.build().to_binary(true).expect("fixture")
    }

    #[test]
    fn from_index_restores_fields_and_removes() {
        let bytes = fixture_bytes();
        let index = crate::MarzIndex::load(&bytes, None).expect("fixture loads");
        let mut builder = MarzBuilder::from_index(&index, None).expect("rehydrates");
        assert_eq!(
            builder.fields(),
            vec!["title".to_string(), "body".to_string()]
        );
        assert_eq!(builder.language(), "en");
        assert_eq!(builder.staged(), 0);
        assert!(builder.remove("a").expect("removes"));
        assert!(!builder.remove("missing").expect("reports absence"));
        assert!(!builder.remove("a").expect("reports second removal"));
        let rebuilt = builder.build_and_load().expect("rebuilds");
        assert_eq!(rebuilt.document_count(), 1);
        // Building again clones the restored state, so the removal survives.
        let again = builder.build_and_load().expect("rebuilds twice");
        assert_eq!(again.document_count(), 1);
    }
}
