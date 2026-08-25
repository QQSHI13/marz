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

use marz_core::languages::{Chinese, English, Japanese, Korean};
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

/// Language codes this build understands, in the order `languages()` reports.
const LANGUAGE_CODES: [&str; 4] = ["en", "zh", "ja", "ko"];

/// Resolve a language code, or explain what the valid ones are.
///
/// The code is checked here rather than deferred, because a typo'd code does not
/// fail — it silently builds an index tokenized by the wrong rules, which then
/// returns no results for queries that look correct.
fn language_for(code: &str) -> PyResult<Arc<dyn Language>> {
    match code {
        "en" => Ok(Arc::new(English)),
        "zh" => Ok(Arc::new(Chinese)),
        "ja" => Ok(Arc::new(Japanese)),
        "ko" => Ok(Arc::new(Korean)),
        other => Err(PyValueError::new_err(format!(
            "unknown language code {other:?}; expected one of {}",
            LANGUAGE_CODES.join(", ")
        ))),
    }
}

/// Language codes this build supports.
#[pyfunction]
fn languages() -> Vec<String> {
    LANGUAGE_CODES.iter().map(|c| (*c).to_string()).collect()
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
        // A mapping raises KeyError for an absent key; treat that as absent
        // rather than propagating, so callers need not pad every document.
        Err(_) => return Ok(None),
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
    // A failure here would mean the exception object rejected an attribute,
    // which cannot happen for a normal exception class; ignore rather than
    // masking the parse error with a second one.
    let _ = value.setattr("start", error.start);
    let _ = value.setattr("end", error.end);
    let _ = value.setattr("query", query);
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
    #[new]
    #[pyo3(signature = (language, *, ref_field = "id", k1 = 1.2, b = 0.75))]
    fn new(language: &str, ref_field: &str, k1: f64, b: f64) -> PyResult<Self> {
        Ok(Self {
            language_code: language.to_string(),
            language: language_for(language)?,
            ref_field: ref_field.to_string(),
            fields: Vec::new(),
            docs: Vec::new(),
            k1,
            b,
        })
    }

    /// Add a searchable field. `boost` multiplies the score of matches in it.
    ///
    /// Fields must be declared before the documents that use them: `add` only
    /// reads the fields declared at the time it is called.
    #[pyo3(signature = (name, boost = 1.0))]
    fn field(&mut self, name: &str, boost: f64) -> PyResult<()> {
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
        self.fields.push((name.to_string(), boost));
        Ok(())
    }

    /// Stage a document for indexing.
    ///
    /// `doc` is any mapping. The reference field must be present and a string;
    /// searchable fields may be absent or `None`.
    #[pyo3(signature = (doc, boost = 1.0))]
    fn add(&mut self, doc: &Bound<'_, PyAny>, boost: f64) -> PyResult<()> {
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
    fn build(&mut self, py: Python<'_>) -> Index {
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
    /// Offsets are in characters, not bytes, so they can index a Python `str`
    /// directly. If the index was built with `positions=False` the terms and
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
    /// data, which is about a tenth of the file.
    #[pyo3(signature = (*, positions = true))]
    fn to_bytes<'py>(&self, py: Python<'py>, positions: bool) -> Bound<'py, PyBytes> {
        let index = self.inner.clone();
        let bytes = py.detach(move || index.to_binary(positions));
        PyBytes::new(py, &bytes)
    }

    /// Serialize to JSON.
    ///
    /// Roughly five times the size of `to_bytes()`. Kept for callers migrating
    /// from a JSON index; prefer `to_bytes()` for anything shipped.
    fn to_json(&self, py: Python<'_>) -> String {
        let index = self.inner.clone();
        py.detach(move || index.to_json())
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
            .map_err(|e| FormatError::new_err(e.to_string()))?
            .language()
            .to_string();
        if let Some(requested) = language {
            if requested != stored {
                return Err(PyValueError::new_err(format!(
                    "index was built for language {stored:?}, not {requested:?}"
                )));
            }
        }
        let lang = language_for(&stored)?;

        // `data` borrows a Python buffer, which cannot cross a GIL release, so
        // copy it first. The copy is a fraction of what the load allocates —
        // materialized postings are several times the encoded bytes — and it
        // buys a concurrent load, which matters when a build script loads
        // per-language indexes in a thread pool.
        let owned = data.to_vec();
        let index = py
            .detach(move || CoreIndex::from_binary(&owned, lang))
            .map_err(|e| FormatError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(index),
            language_code: stored,
        })
    }

    /// Read an index from `to_json()` output.
    #[staticmethod]
    fn from_json(py: Python<'_>, data: &str, language: &str) -> PyResult<Self> {
        let lang = language_for(language)?;
        let data = data.to_string();
        let code = language.to_string();
        let index = py
            .detach(move || CoreIndex::load(&data, lang))
            .map_err(|e| PyValueError::new_err(format!("could not load JSON index: {e}")))?;
        Ok(Self {
            inner: Arc::new(index),
            language_code: code,
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
#[pyfunction]
fn tokenize(text: &str, language: &str) -> PyResult<Vec<String>> {
    let lang = language_for(language)?;
    Ok(lang
        .tokenize(text)
        .into_iter()
        .map(|token| token.term)
        .collect())
}

/// Report what language an index was built for, without loading it.
#[pyfunction]
fn index_language(data: &[u8]) -> PyResult<String> {
    marz_core::BinaryIndex::open(data)
        .map(|index| index.language().to_string())
        .map_err(|e| FormatError::new_err(e.to_string()))
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
    m.add_function(wrap_pyfunction!(index_language, m)?)?;
    Ok(())
}
