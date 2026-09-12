//! Python bindings for Marz.
//!
//! The shape of this API follows how an index is actually used: it is *built*
//! once by a static-site generator running on Python, and *searched* many times
//! in a browser running the WebAssembly build. So the interesting surface here
//! is [`IndexBuilder`] and the two serializers on [`Index`]. Search is exposed
//! as well, because a generator that cannot query what it just built has no way
//! to test it.
//!
//! # Owning data across the boundary
//!
//! Every method that takes text copies it into Rust-owned `String`s before doing
//! any work, rather than holding a `&str` borrowed from a Python object. That
//! costs a copy per field per document, and buys two things: the expensive calls
//! can release the GIL, and no Python object needs to stay alive for the
//! lifetime of the builder.

use std::collections::HashMap;
use std::sync::Arc;

use marz_core::languages::registry;
use marz_core::query_parser::QueryParseError;
use marz_core::{Index as CoreIndex, IndexBuilder as CoreBuilder, Language};
use pyo3::create_exception;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

create_exception!(
    _marz,
    QueryError,
    PyValueError,
    "Raised when a query string cannot be parsed.\n\n\
     Carries `query`, plus `start` and `end` offsets into it."
);

create_exception!(
    _marz,
    FormatError,
    PyValueError,
    "Raised when bytes are not a valid Marz binary index."
);

/// Resolve a language code, warning when it is not one Marz implements.
///
/// A comma-separated list (`"en,ja"`) builds a multi-language configuration
/// running each member's tokenizer. An unknown code does not raise. Marz
/// implements forty-odd languages and the world has more: Vietnamese, Hebrew
/// and Ukrainian all tokenize correctly on whitespace, they simply have no
/// stemmer, so refusing them would block working languages to catch typos.
/// See `marz_core::languages::registry`.
///
/// A typo must not be silent either, so a fallback emits a `UserWarning` naming
/// the code. A build script's output shows `unknown language code 'engish'`
/// rather than quietly producing an index with no stemming.
fn language_for(py: Python<'_>, code: &str) -> PyResult<Arc<dyn Language>> {
    let resolved = registry::resolve_multi(code);
    if !resolved.exact {
        // Name the offending members, not the whole list: `"en,engish"`
        // builds a working multi index whose `engish` member merely lacks a
        // stemmer, which is a different fact from "everything is generic".
        // A lone code keeps the historical message verbatim.
        let parts = registry::canonical_parts(code);
        let detail = if parts.len() <= 1 {
            format!("{code:?}")
        } else {
            let unknown: Vec<&str> = parts
                .iter()
                .filter(|part| !registry::is_supported(part))
                .map(|part| part.as_str())
                .collect();
            if unknown.is_empty() {
                format!("{code:?}")
            } else {
                format!("{} in {code:?}", unknown.join(", "))
            }
        };
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyUserWarning>(),
            &std::ffi::CString::new(format!(
                "unknown language code {detail}: indexing with generic \
                 whitespace tokenization and no stemming. \
                 Call marz.languages() for the codes with full support."
            ))?,
            1,
        )?;
    }
    Ok(resolved.language)
}

/// Resolve without warning, for the load path.
///
/// Stored fallback codes (`vi`, `he`, …) are legitimate working indexes, not
/// typos: warning on every `from_bytes` would spam build scripts that load
/// per-language indexes in a loop. Stored multi-language codes (`en,ja`)
/// resolve back to the same combination.
fn language_for_load(code: &str) -> Arc<dyn Language> {
    registry::resolve_multi(code).language
}

/// Language codes this build supports.
#[pyfunction]
fn languages() -> Vec<String> {
    registry::codes()
        .into_iter()
        .map(|c| c.to_string())
        .collect()
}

/// Pull a field's text out of a document mapping.
///
/// A missing key and an explicit `None` both mean "this document has no such
/// field", which is normal — an optional summary, say. A key present with a
/// non-string value is a bug in the caller's data, so it raises rather than
/// being coerced: stringifying an integer id or a nested dict would index
/// `"{'a': 1}"` as searchable text and nobody would notice until a search
/// missed.
fn field_text(doc: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<String>> {
    let value = match doc.get_item(name) {
        Ok(value) => value,
        // Only a missing key means "absent". Any other failure — e.g. `doc`
        // is not a mapping at all (`doc=42`, `doc="str"`) raising TypeError —
        // must propagate, otherwise every type error becomes a misleading
        // "missing reference field".
        Err(e) => {
            if e.is_instance_of::<pyo3::exceptions::PyKeyError>(doc.py()) {
                return Ok(None);
            }
            return Err(e);
        }
    };
    if value.is_none() {
        return Ok(None);
    }
    value.extract::<String>().map(Some).map_err(|_| {
        PyTypeError::new_err(format!(
            "field {name:?} must be a str or None, got {}",
            value
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "?".to_string())
        ))
    })
}

/// A document staged for indexing, already copied out of Python.
struct StagedDoc {
    doc_ref: String,
    boost: f64,
    fields: HashMap<String, String>,
}

/// Build a `QueryError` whose message shows where in the query the fault is.
///
/// The offsets go on as attributes rather than into the exception's args,
/// because args are what `str(exc)` renders: a three-element args tuple prints
/// as `('unrecognised field', 0, 5)` and buries the sentence a caller needs to
/// read. Attributes keep `str(exc)` a sentence and still let a caller that
/// wants to underline the fault find it.
fn query_error(py: Python<'_>, query: &str, error: &QueryParseError) -> PyErr {
    let err = QueryError::new_err(format!("{} in query {query:?}", error.message));
    let value = err.value(py);
    // Attribute assignment on a freshly created exception cannot fail in
    // practice; expect loudly rather than returning an error that silently
    // lacks its documented span.
    value
        .setattr("start", error.start)
        .expect("exception attribute assignment");
    value
        .setattr("end", error.end)
        .expect("exception attribute assignment");
    value
        .setattr("query", query)
        .expect("exception attribute assignment");
    err
}

/// Builds a search index.
///
/// Configure the reference field and the searchable fields, add documents, then
/// call `build()`.
///
/// ```python
/// b = marz.IndexBuilder("zh", ref_field="location")
/// b.field("title", 10.0)
/// b.field("text")
/// b.add({"location": "guide/intro", "title": "入门", "text": "..."})
/// data = b.build().to_bytes()
/// ```
#[pyclass]
pub struct IndexBuilder {
    language_code: String,
    language: Arc<dyn Language>,
    ref_field: String,
    fields: Vec<(String, f64)>,
    docs: Vec<StagedDoc>,
    k1: f64,
    b: f64,
}

#[pymethods]
impl IndexBuilder {
    /// Create a builder for `language`, one of the codes `languages()` returns.
    ///
    /// `k1` and `b` are the BM25 tuning parameters; the defaults match lunr.
    /// Out-of-range values are clamped, not rejected: `b` to `[0, 1]`, `k1`
    /// to `>= 0` (a negative `k1` keeps the default `1.2`). Only non-finite
    /// values raise.
    #[new]
    #[pyo3(signature = (language, *, ref_field = "id", k1 = 1.2, b = 0.75))]
    fn new(py: Python<'_>, language: &str, ref_field: &str, k1: f64, b: f64) -> PyResult<Self> {
        if language.trim().is_empty() {
            return Err(PyValueError::new_err("language must not be empty"));
        }
        if ref_field.trim().is_empty() {
            return Err(PyValueError::new_err("ref_field must not be empty"));
        }
        if !k1.is_finite() {
            return Err(PyValueError::new_err("k1 must be a finite number"));
        }
        if !b.is_finite() {
            return Err(PyValueError::new_err("b must be a finite number"));
        }
        // Canonical resolved code (`"en, ja"` → `"en,ja"`), matching what
        // the index header stores, so identity survives a roundtrip.
        let language = language_for(py, language)?;
        let language_code = language.code().to_string();
        Ok(Self {
            language_code,
            language,
            ref_field: ref_field.trim().to_string(),
            fields: Vec::new(),
            docs: Vec::new(),
            k1,
            b,
        })
    }

    /// Add a searchable field. `boost` multiplies the score of matches in it.
    ///
    /// Fields must be declared before the documents that use them: `add` only
    /// reads the fields declared at the time it is called. A negative `boost`
    /// is clamped to `0.0` at declaration (it still matches, contributing no
    /// score); only a non-finite boost raises.
    ///
    /// Surrounding whitespace is not part of a name: it is trimmed before
    /// storing, so `field(" title ")` and `title:q` meet.
    #[pyo3(signature = (name, boost = 1.0))]
    fn field(&mut self, name: &str, boost: f64) -> PyResult<()> {
        let name = name.trim();
        if name.is_empty() {
            return Err(PyValueError::new_err("field name must not be empty"));
        }
        if name.contains('/') {
            return Err(PyValueError::new_err(format!(
                "field {name:?} must not contain '/' (breaks FieldRef round-trip)"
            )));
        }
        if name
            .chars()
            .any(|c| c.is_whitespace() || self.language.separator_chars().contains(c))
        {
            return Err(PyValueError::new_err(format!(
                "field {name:?} contains a separator and is unqueryable via field: syntax"
            )));
        }
        if name.chars().any(|c| matches!(c, ':' | '^' | '~' | '\\'))
            || matches!(name.chars().next(), Some('+' | '-'))
        {
            return Err(PyValueError::new_err(format!(
                "field {name:?} contains a query operator and is unqueryable \
                 via field:term syntax"
            )));
        }
        if !boost.is_finite() {
            return Err(PyValueError::new_err("boost must be a finite number"));
        }
        if self.fields.iter().any(|(existing, _)| existing == name) {
            return Err(PyValueError::new_err(format!(
                "field {name:?} is already declared"
            )));
        }
        if name == self.ref_field {
            // Indexing the reference field is legal in lunr but almost always a
            // mistake: it makes every document match its own id, which inflates
            // scores in a way that is hard to trace back to this line.
            return Err(PyValueError::new_err(format!(
                "field {name:?} is the reference field; \
                 pass a different ref_field to index it as text"
            )));
        }
        self.fields.push((name.to_string(), boost.max(0.0)));
        Ok(())
    }

    /// Stage a document for indexing.
    ///
    /// `doc` is any mapping. The reference field must be present and a string;
    /// searchable fields may be absent or `None`. Adding the same reference
    /// twice replaces the previous document (upsert): old postings are dropped
    /// and the document count is not incremented. A negative `boost` is clamped
    /// to `0.0` at declaration; only a non-finite boost raises.
    #[pyo3(signature = (doc, boost = 1.0))]
    fn add(&mut self, doc: &Bound<'_, PyAny>, boost: f64) -> PyResult<()> {
        if !boost.is_finite() {
            return Err(PyValueError::new_err("boost must be a finite number"));
        }
        let boost = boost.max(0.0);
        if self.fields.is_empty() {
            return Err(PyValueError::new_err(
                "declare at least one field with field() before adding documents",
            ));
        }
        let doc_ref = field_text(doc, &self.ref_field)?.ok_or_else(|| {
            PyValueError::new_err(format!(
                "document is missing its reference field {:?}",
                self.ref_field
            ))
        })?;
        if doc_ref.is_empty() {
            return Err(PyValueError::new_err(
                "document reference must not be empty",
            ));
        }

        let mut fields = HashMap::with_capacity(self.fields.len());
        for (name, _) in &self.fields {
            if let Some(text) = field_text(doc, name)? {
                fields.insert(name.clone(), text);
            }
        }
        self.docs.push(StagedDoc {
            doc_ref,
            boost,
            fields,
        });
        Ok(())
    }

    /// Stage many documents. Equivalent to `add` in a loop.
    ///
    /// Not atomic, for the same reason: if a later document raises, the earlier
    /// ones stay staged. Call `clear()` if a partially-consumed builder is not
    /// what you want.
    #[pyo3(signature = (docs, boost = 1.0))]
    fn add_many(&mut self, docs: &Bound<'_, PyAny>, boost: f64) -> PyResult<()> {
        for doc in docs.try_iter()? {
            self.add(&doc?, boost)?;
        }
        Ok(())
    }

    /// Tokenize and score the staged documents.
    ///
    /// This is the expensive call, and it releases the GIL: the documents were
    /// copied into Rust by `add`, so nothing here touches a Python object.
    ///
    /// The builder keeps its staged documents, so calling this twice returns two
    /// equivalent indexes rather than an index and an empty one. Building is not
    /// cheap, but a `build()` that quietly emptied the builder would turn a
    /// stray second call into a silently empty search index.
    fn build(&self, py: Python<'_>) -> Index {
        let language = self.language.clone();
        let ref_field = self.ref_field.clone();
        let fields = self.fields.clone();
        let docs = &self.docs;
        let (k1, b) = (self.k1, self.b);

        let index = py.detach(move || {
            let mut builder = CoreBuilder::new(language);
            builder.ref_field(ref_field).k1(k1).b(b);
            for (name, boost) in &fields {
                builder.field(name.clone(), *boost);
            }
            for doc in docs {
                builder.add(doc.doc_ref.clone(), doc.boost, |name| {
                    doc.fields.get(name).cloned()
                });
            }
            builder.build()
        });

        Index {
            inner: Arc::new(index),
            language_code: self.language_code.clone(),
        }
    }

    /// Discard the staged documents, keeping the field configuration.
    fn clear(&mut self) {
        self.docs.clear();
    }

    /// Number of documents staged and ready to build.
    #[getter]
    fn staged(&self) -> usize {
        self.docs.len()
    }

    /// Declared field names, in declaration order.
    #[getter]
    fn fields(&self) -> Vec<String> {
        self.fields.iter().map(|(name, _)| name.clone()).collect()
    }

    /// The configured reference field.
    #[getter]
    fn ref_field(&self) -> &str {
        &self.ref_field
    }

    /// The configured language code.
    #[getter]
    fn language(&self) -> &str {
        &self.language_code
    }

    fn __repr__(&self) -> String {
        format!(
            "IndexBuilder(language={:?}, ref_field={:?}, fields={:?}, staged={})",
            self.language_code,
            self.ref_field,
            self.fields().as_slice(),
            self.docs.len()
        )
    }
}

/// One search hit.
///
/// Named `Hit` in Rust and `Result` in Python: the Python name is what a caller
/// reads in a loop over `search()`, while `Result` in Rust would shadow
/// `std::result::Result` in every signature in this file.
#[pyclass(frozen, name = "Result")]
pub struct Hit {
    /// The matched document's reference.
    #[pyo3(get)]
    r#ref: String,
    /// BM25 relevance score. Higher is better; only comparable within one
    /// result set.
    #[pyo3(get)]
    score: f64,
    matches: HashMap<String, HashMap<String, Vec<(usize, usize)>>>,
}

#[pymethods]
impl Hit {
    /// Where each matched term occurred, as
    /// `{term: {field: [(start, length), ...]}}`.
    ///
    /// Offsets are in characters, not bytes — but into the **normalized**
    /// field text, not the original: normalization folds full-width Latin,
    /// composes half-width katakana and lowercases, and is not
    /// length-preserving (`ｶﾞ` is two code points becoming one `ガ`), so
    /// every offset after such a character is shifted relative to the input.
    /// Call `marz.normalize(field_text)` first and highlight into that.
    /// If the index was built with `positions=False` the terms and
    /// fields are still reported and only the position lists are empty — enough
    /// to say a match was in the title, not enough to highlight it.
    #[getter]
    fn matches<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (term, fields) in &self.matches {
            let per_field = PyDict::new(py);
            for (field, positions) in fields {
                let items: Vec<Bound<'py, PyTuple>> = positions
                    .iter()
                    .map(|(start, length)| PyTuple::new(py, [*start, *length]))
                    .collect::<PyResult<_>>()?;
                per_field.set_item(field, PyList::new(py, items)?)?;
            }
            out.set_item(term, per_field)?;
        }
        Ok(out)
    }

    /// The matched terms, sorted.
    ///
    /// For CJK these are the bigrams the query tokenized into, not words.
    #[getter]
    fn terms(&self) -> Vec<String> {
        let mut terms: Vec<String> = self.matches.keys().cloned().collect();
        terms.sort();
        terms
    }

    fn __repr__(&self) -> String {
        format!("Result(ref={:?}, score={})", self.r#ref, self.score)
    }
}

/// A built search index.
#[pyclass(frozen)]
pub struct Index {
    // Arc so the GIL can be released around a search without cloning the index.
    inner: Arc<CoreIndex>,
    language_code: String,
}

#[pymethods]
impl Index {
    /// Search the index, returning hits in descending score order.
    ///
    /// Query syntax: bare terms, `+required`, `-prohibited`, `field:term`,
    /// `term*` wildcards, `term~N` fuzzy matching, and `^N` term boosts.
    /// Raises `QueryError` if the query cannot be parsed.
    fn search(&self, py: Python<'_>, query: &str) -> PyResult<Vec<Hit>> {
        let index = self.inner.clone();
        let owned = query.to_string();
        let hits = py.detach(move || index.search(&owned));
        let hits = match hits {
            Ok(hits) => hits,
            Err(e) => return Err(query_error(py, query, &e)),
        };
        Ok(hits
            .into_iter()
            .map(|hit| Hit {
                r#ref: hit.ref_id,
                score: hit.score,
                matches: hit.match_data.terms,
            })
            .collect())
    }

    /// Serialize to the compact binary format.
    ///
    /// Pass `positions=False` to drop highlighting and CJK phrase-verification
    /// data, which is about a tenth of the file. Note this also disables the
    /// CJK phrase ranking boost: search still works, but an exact phrase no
    /// longer outranks scattered bigrams.
    #[pyo3(signature = (*, positions = true))]
    fn to_bytes<'py>(&self, py: Python<'py>, positions: bool) -> Bound<'py, PyBytes> {
        let index = self.inner.clone();
        let bytes = py.detach(move || index.to_binary(positions));
        PyBytes::new(py, &bytes)
    }

    /// Read an index from `to_bytes()` output.
    ///
    /// `language` must match the language it was built with — the index stores
    /// the code, and a mismatch is rejected rather than silently tokenizing
    /// queries by rules that disagree with the indexed terms.
    ///
    /// This materializes the postings rather than reading them in place, so it
    /// costs time and memory proportional to the index.
    #[staticmethod]
    #[pyo3(signature = (data, language = None))]
    fn from_bytes(py: Python<'_>, data: &[u8], language: Option<&str>) -> PyResult<Self> {
        // Read the header first, so a wrong language code is reported before
        // spending the load, and so a caller gets a FormatError rather than a
        // confusing language mismatch on bytes that are not an index at all.
        let stored = marz_core::BinaryIndex::open(data)
            .map_err(|e| FormatError::new_err(format!("not a Marz index: {e}")))?
            .language()
            .to_string();
        if let Some(requested) = language {
            // Canonical member lists, not raw strings: `"en, ja"` and
            // `"en,ja"` are the same configuration (member order still
            // matters — it affects stemming — so no sorting).
            if registry::canonical_parts(requested) != registry::canonical_parts(&stored) {
                return Err(FormatError::new_err(format!(
                    "index was built for language {stored:?}, not {requested:?}"
                )));
            }
        }
        // Silent resolve: stored fallback codes are legitimate, not typos.
        let lang = language_for_load(&stored);

        // `data` borrows a Python buffer, which cannot cross a GIL release, so
        // copy it first. The copy is a fraction of what the load allocates —
        // materialized postings are several times the encoded bytes — and it
        // buys a concurrent load, which matters when a build script loads
        // per-language indexes in a thread pool.
        let owned = data.to_vec();
        let index = py
            .detach(move || CoreIndex::from_binary(&owned, lang))
            .map_err(|e| FormatError::new_err(format!("not a Marz index: {e}")))?;
        Ok(Self {
            inner: Arc::new(index),
            language_code: stored,
        })
    }

    /// Indexed field names.
    #[getter]
    fn fields(&self) -> Vec<String> {
        self.inner.fields().to_vec()
    }

    /// Number of indexed documents.
    #[getter]
    fn document_count(&self) -> usize {
        self.inner.document_count()
    }

    /// Number of distinct indexed terms.
    #[getter]
    fn term_count(&self) -> usize {
        self.inner.term_count()
    }

    /// The language code this index was built with.
    #[getter]
    fn language(&self) -> &str {
        &self.language_code
    }

    fn __len__(&self) -> usize {
        self.inner.document_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "Index(language={:?}, documents={}, terms={})",
            self.language_code,
            self.inner.document_count(),
            self.inner.term_count()
        )
    }
}

/// Tokenize `text` the way the index would, for inspecting how a query splits.
///
/// Useful for understanding CJK results: `tokenize("検索エンジン", "ja")` shows
/// the overlapping bigrams that are actually indexed.
///
/// This is a pre-pipeline split: trimming, stop-word removal and stemming are
/// not applied, so the terms shown are not always the terms indexed (e.g.
/// English shows `running`, the index holds `run`).
#[pyfunction]
fn tokenize(py: Python<'_>, text: &str, language: &str) -> PyResult<Vec<String>> {
    let lang = language_for(py, language)?;
    Ok(lang
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
/// length-preserving, so highlight `marz.normalize(field_text)` and never the
/// raw field.
///
/// `language` selects the lowercasing rules and must be the index's language:
/// Turkish (`"tr"`) folds `I` to `ı`, every other language to `i`. A
/// multi-language code containing Turkish folds the default way instead —
/// no single folding serves both members, so dotted-capital-I offsets in
/// Turkish documents may shift by a character; search itself is unaffected
/// (each member folds its own way at query time). Unknown codes warn and
/// fall back exactly like `tokenize`.
#[pyfunction]
#[pyo3(signature = (text, language = "en"))]
fn normalize(py: Python<'_>, text: &str, language: &str) -> PyResult<String> {
    // Warn on unknown codes exactly like `tokenize` does; the fold itself
    // dispatches on the trimmed code below.
    let _ = language_for(py, language)?;
    Ok(marz_core::normalize::normalize_for_language(
        language.trim(),
        text,
    ))
}

/// Report what language an index was built for, without loading it.
#[pyfunction]
fn index_language(data: &[u8]) -> PyResult<String> {
    marz_core::BinaryIndex::open(data)
        .map(|index| index.language().to_string())
        .map_err(|e| FormatError::new_err(format!("not a Marz index: {e}")))
}

/// Native extension module. Import from `marz`, not from here.
#[pymodule]
fn _marz(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<IndexBuilder>()?;
    m.add_class::<Index>()?;
    m.add_class::<Hit>()?;
    m.add("QueryError", m.py().get_type::<QueryError>())?;
    m.add("FormatError", m.py().get_type::<FormatError>())?;
    m.add_function(wrap_pyfunction!(languages, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(index_language, m)?)?;
    Ok(())
}
