//! Index builder and searcher.
//!
//! # Scoring: query-time BM25, not precomputed vectors
//!
//! lunr precomputes a *field vector* for every (document, field): a sparse
//! vector of BM25 weights, one entry per term in that field. At query time it
//! builds a query vector and takes an asymmetric cosine similarity against
//! each field vector.
//!
//! Marz scores at query time instead, accumulating BM25 contributions per
//! matching document. The two produce the same ranking, because the weights
//! being summed are identical — but the stored form is much cheaper:
//!
//! * **Size.** Field vectors duplicate the entire inverted index as
//!   `f64` weights. Measured on Chinese Wikipedia text they were a flat
//!   16% of the serialized index, and their cost grows with the corpus.
//!   Query-time scoring stores one `u32` term frequency per posting instead.
//! * **Boosts.** Field and document boosts configured at build time are still
//!   baked in — same as lunr. What query-time scoring buys is that the
//!   *clause* boost (`term^N`) applies at search time, so one index serves
//!   differently-weighted queries without rebuilding.
//! * **Zero-copy.** A sparse `f64` vector per document-field cannot be read
//!   from a memory-mapped byte slice without allocating. Term frequencies and
//!   field lengths can.
//!
//! Only three things are needed at query time to compute BM25: the term's
//! document frequency (the posting list length), the term frequency in the
//! field (stored per posting), and the field length plus its corpus average
//! (stored per document-field). All three are small integers.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::language::LanguageRef;
use crate::normalize::{normalize_for_language, normalize_tr};
use crate::phrase::{extract_phrases, Phrase, VerificationCache, PHRASE_BOOST};
use crate::pipeline::Pipeline;
use crate::query::{Presence, Query};
use crate::query_parser::{parse_query, QueryParseError};
use crate::token_set::TokenSet;
use crate::{bm25_weight, idf};

/// A reference to one field of one document.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct FieldRef {
    /// Document reference.
    doc_ref: String,
    /// Name of the indexed field.
    field_name: String,
}

impl FieldRef {
    /// Create a new field reference.
    pub fn new(doc_ref: impl Into<String>, field_name: impl Into<String>) -> Self {
        Self {
            doc_ref: doc_ref.into(),
            field_name: field_name.into(),
        }
    }

    /// Return the document reference.
    pub fn doc_ref(&self) -> &str {
        &self.doc_ref
    }

    /// Return the field name.
    pub fn field_name(&self) -> &str {
        &self.field_name
    }

    /// Parse a field reference from its `fieldName/docRef` representation.
    ///
    /// Splits on the *first* `/`, so a document reference may itself contain
    /// slashes — which is normal, since document references are usually URL
    /// paths. Returns `None` if there is no separator.
    ///
    /// Note the order: field name first. An earlier version split on the last
    /// `/` and treated the leading part as the document reference, which both
    /// inverted the two halves and truncated any docref containing a slash.
    pub fn from_string(s: &str) -> Option<Self> {
        let (field_name, doc_ref) = s.split_once('/')?;
        Some(Self {
            doc_ref: doc_ref.to_string(),
            field_name: field_name.to_string(),
        })
    }
}

impl std::fmt::Display for FieldRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.field_name, self.doc_ref)
    }
}

/// Per-document posting data for one term in one field.
#[derive(Debug, Clone, Default)]
pub struct PostingDoc {
    /// How many times the term occurs in this document field.
    ///
    /// This is BM25's `tf`. It is stored rather than derived from
    /// `positions.len()` so that positions can live in a separate, lazily
    /// loaded section of the binary format without costing scoring accuracy.
    pub term_frequency: u32,
    /// Term positions in the field, as `(start, length)` character offsets.
    ///
    /// Used for highlighting and for CJK phrase verification.
    pub positions: Vec<(usize, usize)>,
}

/// Inverted-index posting for a single term.
#[derive(Debug, Clone, Default)]
pub struct Posting {
    /// Maps field name -> document reference -> posting data.
    pub fields: HashMap<String, HashMap<String, PostingDoc>>,
}

impl Posting {
    /// Number of distinct documents containing this term, across all fields.
    ///
    /// This is BM25's `df`. A document containing the term in two fields is
    /// counted once, which is what makes IDF a property of the term rather
    /// than of the term-field pair.
    pub fn document_frequency(&self) -> usize {
        let mut docs: HashSet<&str> = HashSet::new();
        for field_docs in self.fields.values() {
            docs.extend(field_docs.keys().map(String::as_str));
        }
        docs.len()
    }
}

/// A matching term/field combination returned with a search result.
#[derive(Debug, Clone, Default)]
pub struct MatchData {
    /// Maps a matched term to a map of fields -> positions.
    pub terms: HashMap<String, HashMap<String, Vec<(usize, usize)>>>,
}

impl MatchData {
    /// Record that `term` matched in `field` at the given positions.
    ///
    /// Positions are deduplicated. Wildcard and fuzzy expansion can route the
    /// same indexed term through a clause more than once, and without this the
    /// same position is reported repeatedly.
    pub fn add_term(&mut self, term: &str, field: &str, positions: &[(usize, usize)]) {
        // `entry` would need an owned key to ask the question, so it allocates a
        // `String` for the term and another for the field on every call and drops
        // both again whenever the key is already present — which, in a query that
        // scores a term across thousands of documents, is almost every call. The
        // extra lookup is cheaper than the allocation it replaces.
        if !self.terms.contains_key(term) {
            self.terms.insert(term.to_string(), HashMap::new());
        }
        let fields = self.terms.get_mut(term).expect("just inserted");
        if !fields.contains_key(field) {
            fields.insert(field.to_string(), Vec::new());
        }
        let entry = fields.get_mut(field).expect("just inserted");
        if positions.is_empty() {
            return;
        }
        // Extend + sort + dedup instead of O(n²) `contains` per position.
        // Posting positions are already ascending, so this is usually one
        // linear pass; worst case is O(n log n), never quadratic.
        entry.extend_from_slice(positions);
        entry.sort_unstable();
        entry.dedup();
    }

    /// Merge another match data object into this one.
    pub fn merge(&mut self, other: &Self) {
        for (term, fields) in &other.terms {
            let entry = self.terms.entry(term.clone()).or_default();
            for (field, positions) in fields {
                let target = entry.entry(field.clone()).or_default();
                if positions.is_empty() {
                    continue;
                }
                target.extend_from_slice(positions);
                target.sort_unstable();
                target.dedup();
            }
        }
    }
}

/// A single search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Document reference.
    pub ref_id: String,
    /// Relevance score.
    pub score: f64,
    /// Match metadata.
    pub match_data: MatchData,
}

#[derive(Debug, Clone)]
struct FieldConfig {
    name: String,
    boost: f64,
}

/// Corpus statistics needed to score BM25 at query time.
#[derive(Debug, Clone, Default)]
struct Stats {
    /// Total number of documents in the index.
    document_count: usize,
    /// Length in tokens of each document field: `doc_ref -> field_name -> len`.
    ///
    /// Nested (not `HashMap<FieldRef, _>`) so scoring looks up
    /// `get(doc).get(field)` with borrowed `&str` and zero allocation in the
    /// hot loop.
    field_lengths: HashMap<String, HashMap<String, usize>>,
    /// Mean field length per field name, over documents that have the field
    /// non-empty. Empty/missing fields are excluded (they would drag the
    /// average toward zero and corrupt length normalization).
    average_field_lengths: HashMap<String, f64>,
    /// Per-field boosts configured at build time.
    field_boosts: HashMap<String, f64>,
    /// Per-document boosts configured at build time.
    doc_boosts: HashMap<String, f64>,
    /// BM25 term-frequency saturation parameter.
    k1: f64,
    /// BM25 length-normalization parameter.
    b: f64,
}

/// Index builder.
///
/// Holds the full inverted index in memory: comfortable into the tens of
/// thousands of documents (the docs-site scale this engine targets), not
/// millions. Staged documents are owned twice over (once staged, once
/// indexed), and re-adding a reference scans every posting list, so bulk
/// loads should prefer fresh references over upserts.
#[derive(Clone)]
pub struct IndexBuilder {
    language: LanguageRef,
    ref_field: String,
    fields: Vec<FieldConfig>,
    field_lengths: HashMap<String, HashMap<String, usize>>,
    inverted_index: BTreeMap<String, Posting>,
    document_count: usize,
    doc_boosts: HashMap<String, f64>,
    k1: f64,
    b: f64,
}

impl IndexBuilder {
    /// Create a new builder for the given language.
    pub fn new(language: LanguageRef) -> Self {
        Self {
            language,
            ref_field: "id".to_string(),
            fields: Vec::new(),
            field_lengths: HashMap::new(),
            inverted_index: BTreeMap::new(),
            document_count: 0,
            doc_boosts: HashMap::new(),
            k1: 1.2,
            b: 0.75,
        }
    }

    /// Set the document reference field. Defaults to `"id"`.
    pub fn ref_field(&mut self, field: impl Into<String>) -> &mut Self {
        // Trimmed like field names and the bindings: padding is formatting.
        self.ref_field = field.into().trim().to_string();
        self
    }

    /// Return the configured document reference field.
    pub fn ref_field_name(&self) -> &str {
        &self.ref_field
    }

    /// Declared fields in declaration order, with their boosts.
    ///
    /// The bindings stage field names separately from the core builder, so a
    /// rehydrated builder restores that list from here rather than losing the
    /// boosts by re-declaring names alone.
    pub fn declared_fields(&self) -> Vec<(String, f64)> {
        self.fields
            .iter()
            .map(|f| (f.name.clone(), f.boost))
            .collect()
    }

    /// Add a field to the index. `boost` defaults to `1.0`.
    ///
    /// Declaring the same name twice keeps the first declaration and ignores
    /// the second (no double-counting). The Python and WASM bindings reject
    /// duplicates with an error instead — same outcome for well-formed
    /// configurations, louder for mistakes.
    ///
    /// Panics on an empty name, one containing `/`, one containing a
    /// separator character, or one containing a query operator (`:` `^` `~`
    /// `\`, or leading `+`/`-`): a `/` breaks `FieldRef` round-tripping, an
    /// empty name is always a configuration bug, and a name the query lexer
    /// would split is unrepresentable in `field:term` syntax, so declaring it
    /// can only build an unqueryable field. Failing fast beats a silently
    /// corrupt index.
    /// Non-finite boosts fall back to `1.0`; negative boosts clamp to `0.0`
    /// (they still match, contributing no score).
    pub fn field(&mut self, name: impl Into<String>, boost: f64) -> &mut Self {
        // Trimmed like the bindings do: `" title "` and `title:q` must meet.
        let name = name.into();
        let name = name.trim().to_string();
        assert!(!name.is_empty(), "field name must not be empty");
        assert!(
            !name
                .chars()
                .any(|c| c.is_whitespace() || self.language.separator_chars().contains(c)),
            "field name {name:?} contains a separator and is unqueryable"
        );
        assert!(
            !name.chars().any(|c| matches!(c, ':' | '^' | '~' | '\\'))
                && !matches!(name.chars().next(), Some('+' | '-')),
            "field name {name:?} contains a query operator and is unqueryable \
             via field:term syntax"
        );
        debug_assert!(
            name != self.ref_field,
            "indexing the reference field inflates scores; the bindings reject this"
        );
        assert!(
            !name.contains('/'),
            "field name {name:?} must not contain '/' (breaks FieldRef round-trip)"
        );
        if self.fields.iter().any(|f| f.name == name) {
            // Deduplicate: pushing twice would double-count `tf` at scoring time
            // via duplicated field boosts. Keep the first declaration.
            return self;
        }
        let boost = if boost.is_finite() {
            boost.max(0.0)
        } else {
            1.0
        };
        self.fields.push(FieldConfig { name, boost });
        self
    }

    /// Set the BM25 `k1` parameter. Non-finite or negative values keep `1.2`.
    pub fn k1(&mut self, k1: f64) -> &mut Self {
        if k1.is_finite() && k1 >= 0.0 {
            self.k1 = k1;
        }
        self
    }

    /// Set the BM25 `b` parameter, clamped to `[0, 1]`. Non-finite keeps `0.75`.
    pub fn b(&mut self, b: f64) -> &mut Self {
        if b.is_finite() {
            self.b = b.clamp(0.0, 1.0);
        }
        self
    }

    /// Rehydrate a builder from a built [`Index`] for incremental updates.
    ///
    /// Moves the inverted index, field lengths, document boosts, document
    /// count, field names/boosts and BM25 parameters back into a builder, so a
    /// caller can [`remove`](Self::remove) or re-[`add`](Self::add) documents
    /// and [`build`](Self::build) again without re-adding the whole corpus.
    /// The on-disk format is never patched in place: rebuild and rewrite the
    /// bytes instead (see [`Index::to_binary`]).
    ///
    /// The document reference field name is not stored in the index, so the
    /// builder resets to `"id"`: call [`ref_field`](Self::ref_field) afterwards
    /// if the corpus uses another name. Averages are re-derived on `build`,
    /// so scoring is unchanged.
    pub fn from_index(index: Index) -> Self {
        let language = index.pipeline.language();
        let Index {
            inverted_index,
            token_set: _,
            fields,
            stats,
            pipeline: _,
        } = index;
        let Stats {
            document_count,
            field_lengths,
            average_field_lengths: _,
            field_boosts,
            doc_boosts,
            k1,
            b,
        } = stats;
        let field_configs = fields
            .into_iter()
            .map(|name| {
                let boost = field_boosts.get(&name).copied().unwrap_or(1.0);
                FieldConfig { name, boost }
            })
            .collect();
        Self {
            language,
            ref_field: "id".to_string(),
            fields: field_configs,
            field_lengths,
            inverted_index,
            document_count,
            doc_boosts,
            k1,
            b,
        }
    }

    /// Add a document to the index.
    ///
    /// `field_getter` receives a field name and should return the raw text for
    /// that field, or `None` if the field is missing.
    ///
    /// Adding the same `doc_ref` twice **replaces** the previous document
    /// (upsert): old postings are removed and `document_count` is not
    /// incremented. Merging would double `tf` while overwriting field lengths,
    /// corrupting BM25 statistics.
    pub fn add<F>(&mut self, doc_ref: impl Into<String>, doc_boost: f64, mut field_getter: F)
    where
        F: FnMut(&str) -> Option<String>,
    {
        let doc_ref = doc_ref.into();
        assert!(
            !doc_ref.is_empty(),
            "empty document references are rejected at the binding boundary"
        );
        let doc_boost = if doc_boost.is_finite() {
            doc_boost.max(0.0)
        } else {
            1.0
        };
        let is_update = self.doc_boosts.contains_key(&doc_ref);
        if is_update {
            self.remove_doc(&doc_ref);
        } else {
            self.document_count += 1;
        }
        self.doc_boosts.insert(doc_ref.clone(), doc_boost);

        let pipeline = Pipeline::new(self.language.clone());

        for field in &self.fields {
            let text = field_getter(&field.name).unwrap_or_default();
            let tokens = self.language.tokenize(&text);
            let terms = pipeline.run_index(tokens);

            self.field_lengths
                .entry(doc_ref.clone())
                .or_default()
                .insert(field.name.clone(), terms.len());

            for token in terms {
                let position = token.position().unwrap_or((0, token.term.chars().count()));

                let posting_doc = self
                    .inverted_index
                    .entry(token.term)
                    .or_default()
                    .fields
                    .entry(field.name.clone())
                    .or_default()
                    .entry(doc_ref.clone())
                    .or_default();

                posting_doc.term_frequency += 1;
                posting_doc.positions.push(position);
            }
        }
    }

    /// Remove all postings and lengths for `doc_ref` (upsert helper).
    fn remove_doc(&mut self, doc_ref: &str) {
        for posting in self.inverted_index.values_mut() {
            for docs in posting.fields.values_mut() {
                docs.remove(doc_ref);
            }
            // Drop fields left empty so `document_frequency` and serialization
            // do not see ghost entries.
            posting.fields.retain(|_, docs| !docs.is_empty());
        }
        self.inverted_index.retain(|_, p| !p.fields.is_empty());
        self.field_lengths.remove(doc_ref);
    }

    /// Remove the document `doc_ref` from the index.
    ///
    /// Returns `true` when a document was present and removed, `false` when no
    /// document used that reference. Removing an unknown reference is a no-op.
    /// This is the public form of the upsert helper: [`add`](Self::add) on an
    /// existing reference replaces it, while this deletes it outright and
    /// decrements the document count.
    pub fn remove(&mut self, doc_ref: &str) -> bool {
        if self.doc_boosts.remove(doc_ref).is_none() {
            return false;
        }
        self.remove_doc(doc_ref);
        self.document_count = self.document_count.saturating_sub(1);
        true
    }

    /// Consume the builder and produce a searchable [`Index`].
    pub fn build(self) -> Index {
        let field_names: Vec<String> = self
            .fields
            .iter()
            .map(|f| f.name.clone())
            .collect::<Vec<_>>();
        // One implementation with `from_binary`: the average is a pure function
        // of the length matrix, and a second copy is exactly the kind of thing
        // that drifts when only one side is fixed.
        let average_field_lengths = average_field_lengths(&self.field_lengths, &field_names);
        let token_set = TokenSet::from_strs(self.inverted_index.keys().map(String::as_str));

        let stats = Stats {
            document_count: self.document_count,
            field_lengths: self.field_lengths,
            average_field_lengths,
            field_boosts: self
                .fields
                .iter()
                .map(|f| (f.name.clone(), f.boost))
                .collect(),
            doc_boosts: self.doc_boosts,
            k1: self.k1,
            b: self.b,
        };

        Index {
            inverted_index: self.inverted_index,
            token_set,
            fields: self.fields.iter().map(|f| f.name.clone()).collect(),
            stats,
            pipeline: Pipeline::new(self.language),
        }
    }
}

/// Built search index.
#[derive(Clone)]
pub struct Index {
    inverted_index: BTreeMap<String, Posting>,
    token_set: TokenSet,
    fields: Vec<String>,
    stats: Stats,
    pipeline: Pipeline,
}

/// One term expansion contributing to a clause, with its effective boost.
struct Contribution<'a> {
    term: &'a str,
    posting: &'a Posting,
    boost: f64,
    /// IDF for this term, computed once here rather than per scored document.
    ///
    /// `Posting::document_frequency` builds a `HashSet` over every document in
    /// the posting list, so calling it from the scoring loop made scoring a term
    /// quadratic in the number of documents containing it — 3.2 s for a
    /// two-term query over 5,000 documents, against 74 ms over 1,000. IDF is a
    /// property of the term and the corpus, so nothing in the inner loop can
    /// change it.
    idf: f64,
    /// Fold union this expansion belongs to, if any, as `(union, fold)`.
    ///
    /// Folds of one intent (bypass fold union, or pipeline tokens sharing a
    /// source-word start offset) sum within each fold and take the maximum
    /// across folds, per document-field — instead of summing everything, which
    /// would double-count a document matching both (e.g. `ISTANBUL*` meeting
    /// `istanbul` and `ıstanbul` postings of one word). `None` sums as before.
    fold: Option<(usize, usize)>,
}

/// A clause's terms with their fold identities: `(union, fold)` per term,
/// or `None` for ungrouped terms that sum directly. See `Contribution::fold`.
type ClauseTerms = (Vec<String>, Vec<Phrase>, Vec<Option<(usize, usize)>>);

impl Index {
    /// Search the index using lunr query syntax.
    ///
    /// Supports `+` required, `-` prohibited, `field:term`, `term^boost`,
    /// `term~edits`, and `*` wildcards.
    pub fn search(&self, query_string: &str) -> Result<Vec<SearchResult>, QueryParseError> {
        self.search_limited(query_string, None)
    }

    /// Search, materializing match metadata for at most `limit` results.
    ///
    /// Equivalent to [`Index::search`] followed by truncation, except that
    /// highlighting data is built only for the documents that survive the cut.
    /// The difference is not cosmetic: a wildcard query over a 5,000-document
    /// corpus spent 8.3 s and ~451 MB building 475,000 match-term entries and
    /// 2.3 million positions, all to return the top ten.
    ///
    /// `limit` is a cap, not an offset: results are ranked identically either
    /// way, and the first `N` of a limited search equal the first `N` of an
    /// unlimited one.
    pub fn search_limited(
        &self,
        query_string: &str,
        limit: Option<usize>,
    ) -> Result<Vec<SearchResult>, QueryParseError> {
        let language = self.pipeline.language();
        let separators = language.separator_chars();
        let query = parse_query(query_string, &self.fields, separators, &language)?;
        Ok(self.execute_query(&query, limit))
    }

    /// Execute a programmatic [`Query`] against the index.
    ///
    /// Unlike the string parser (which rejects unknown `field:` scopes with
    /// [`QueryParseError`](crate::query_parser::QueryParseError)), a clause
    /// scoped to a field the index does not have simply matches nothing.
    /// Validate scopes up front if silence would hide a caller bug.
    pub fn query(&self, query: &Query) -> Vec<SearchResult> {
        self.execute_query(query, None)
    }

    /// BM25 score for one term occurrence in one document field.
    ///
    /// `idf_value` is passed in rather than derived from `posting`, because it
    /// depends only on the term and is therefore loop-invariant — see
    /// [`Contribution::idf`].
    fn score_term(
        &self,
        posting: &Posting,
        field_name: &str,
        doc_ref: &str,
        boost: f64,
        idf_value: f64,
    ) -> f64 {
        let Some(posting_doc) = posting
            .fields
            .get(field_name)
            .and_then(|docs| docs.get(doc_ref))
        else {
            return 0.0;
        };

        // Zero-allocation nested lookup: `doc -> field -> len`.
        let field_length = self
            .stats
            .field_lengths
            .get(doc_ref)
            .and_then(|m| m.get(field_name))
            .copied()
            .unwrap_or(0) as f64;
        let average_length = self
            .stats
            .average_field_lengths
            .get(field_name)
            .copied()
            .unwrap_or(0.0);
        // A field with no tokens anywhere in the corpus cannot be scored: the
        // BM25 length-normalization term would divide by zero.
        if average_length == 0.0 {
            return 0.0;
        }

        let field_boost = self
            .stats
            .field_boosts
            .get(field_name)
            .copied()
            .unwrap_or(1.0);
        let doc_boost = self.stats.doc_boosts.get(doc_ref).copied().unwrap_or(1.0);

        let weight = bm25_weight(
            idf_value,
            posting_doc.term_frequency as f64,
            field_length,
            average_length,
            self.stats.k1,
            self.stats.b,
            field_boost,
            doc_boost,
        );
        saturating_product(weight, boost)
    }

    /// Fold bypass terms (pipeline disabled) the way the parser would have.
    ///
    /// Parsed clauses arrive pre-folded; programmatic ones arrive raw. A lone
    /// default fold would additionally lose Turkish in multi-language indexes,
    /// so a `tr` member unions both folds.
    fn bypass_terms(language: &LanguageRef, raw: &str) -> Vec<String> {
        let code = language.code();
        let mut out = vec![normalize_for_language(code, raw)];
        if crate::normalize::uses_raw_passthrough(code) {
            let turkish = normalize_tr(raw);
            if turkish != out[0] {
                out.push(turkish);
            }
        }
        out
    }

    /// Expand a clause term into the set of indexed terms it matches.
    ///
    /// A term containing `*` is always a wildcard, even when `edits` is also
    /// set (`foo*~1`): fuzzy+wildcard is not a defined combination, and scoring
    /// it as fuzzy would silently drop the wildcard. Documented here rather
    /// than rejected so `term*~N` keeps returning wildcard matches.
    ///
    /// Only unescaped stars count: an escaped `\*` is literal text. The exact
    /// path unescapes (`a\*b` looks up indexed `a*b`); the wildcard decision
    /// itself reads the raw clause text, whose retained backslashes still mark
    /// escaped stars.
    fn expand_clause_term(
        &self,
        clause_term: &str,
        term: &str,
        edits: Option<usize>,
    ) -> Vec<String> {
        if crate::query::has_unescaped_wildcard(clause_term) {
            self.expand_wildcard_term(term)
        } else if let Some(edits) = edits {
            self.token_set.expand_fuzzy(term, edits)
        } else {
            self.token_set
                .expand_exact(&crate::query::unescape_term(term))
        }
    }

    /// Expand a wildcard clause against the indexed terms, which are *stems*.
    ///
    /// Three cases need three different treatments; conflating them is how
    /// wildcards quietly lose recall and gain false positives.
    ///
    /// * **A star with literal text on both sides** (`secur*ity`) is matched as
    ///   written. It must not be stemmed: the stemmer treats `*` as an ordinary
    ///   character, and Porter trims `secur*ity` to `secur*`, which then matches
    ///   `secure`.
    /// * **A star at an edge** (`aby*`, `*bar`, `*foo*`) keeps the raw pattern,
    ///   because surface prefixes are what the user means and a stemmed prefix
    ///   is not prefix-stable in either direction: stemming `aby` gives `abi`
    ///   and stops finding `abysses` (indexed as `abyss`), while stemming `abas`
    ///   gives `aba` and starts finding `abacinate`.
    /// * **…plus an exact lookup of the stemmed body**, because the dictionary
    ///   holds `searchabl` for `searchable`, and the raw surface pattern
    ///   `searchable*` is longer than the term it should match.
    ///
    /// The union is what restores prefix recall without reintroducing the
    /// over-stemming false positives.
    fn expand_wildcard_term(&self, term: &str) -> Vec<String> {
        if crate::query::has_interior_wildcard(term) {
            return self.token_set.expand(term);
        }

        let mut out = self.token_set.expand(term);

        // Only a plain word body can be stemmed. An escaped star is literal text
        // (`a\*b`), a stars-only pattern (`**`) has no body, and stemming a
        // non-Latin term would run a per-word stemmer over CJK characters.
        let body = term.trim_matches('*');
        if !body.is_empty() && !term.contains('\\') && !crate::query::is_non_latin_term(body) {
            let stem = self.pipeline.language.stem(body);
            if stem != body {
                out.extend(self.token_set.expand_exact(&stem));
            }
        }
        out
    }

    /// Boost multiplier for `term` in one document field, from phrase matches.
    ///
    /// Multiplicative per verified phrase, per field: two verified phrases give
    /// `PHRASE_BOOST²`. Word-tokenized languages never produce phrases, so they
    /// take the empty-slice fast path and pay nothing.
    fn phrase_boost_for(
        &self,
        phrases: &[Phrase],
        term: &str,
        field_name: &str,
        doc_ref: &str,
        cache: &mut VerificationCache,
    ) -> f64 {
        let mut boost = 1.0;
        for (i, phrase) in phrases.iter().enumerate() {
            if !phrase.contains(term) {
                continue;
            }
            let verified = cache.verify(i, phrase, field_name, doc_ref, |t| {
                self.inverted_index
                    .get(t)
                    .and_then(|posting| posting.fields.get(field_name))
                    .and_then(|docs| docs.get(doc_ref))
                    .map(|posting_doc| posting_doc.positions.as_slice())
            });
            if verified {
                boost = saturating_product(boost, PHRASE_BOOST);
            }
        }
        boost
    }

    fn execute_query(&self, query: &Query, limit: Option<usize>) -> Vec<SearchResult> {
        let language = self.pipeline.language();
        // Keyed by `&str` borrowed from `self.inverted_index`, which outlives the
        // query. Owning these meant a `String` allocation per document per term
        // in each of three maps, and the scoring loop runs once per matched
        // document — so on a query matching the whole corpus the allocator was
        // doing more work than the scorer.
        let mut scores: HashMap<&str, f64> = HashMap::new();
        // Expansions per scored clause, kept so match metadata can be built
        // after ranking instead of while scoring. The clause's field scope rides
        // along: metadata is per (term, field), so replaying a clause without it
        // would report matches in fields the clause never queried, and a
        // `title:` query would highlight body text.
        let mut scored_clauses: Vec<(&[String], Vec<Contribution<'_>>)> = Vec::new();
        let mut required: Option<HashSet<&str>> = None;
        let mut prohibited: HashSet<&str> = HashSet::new();
        let mut matched_any = false;

        for clause in &query.clauses {
            // Phrase verification is per (clause phrase, field, document) but
            // is consulted once per phrase *term*, so the verdict is cached
            // within the clause. The cache lives inside the loop deliberately:
            // phrase indices restart at 0 per clause, so sharing one cache
            // across clauses lets clause N's verdict leak into clause M.
            let mut phrase_cache = VerificationCache::default();
            // Phrases are derived from the pipeline's tokens, so they are only
            // available when the pipeline ran. A wildcard clause disables it,
            // and a wildcard is an explicit request for loose matching anyway.
            // Each term carries an optional (union, fold) identity (see
            // `Contribution::fold`): `None` sums exactly as before; unions of
            // one intent maximize across folds per document-field.
            let (terms, phrases, folds): ClauseTerms = if clause.use_pipeline {
                let tokens = self.pipeline.run_search(&clause.term);
                let phrases = extract_phrases(&tokens, &language);
                let terms: Vec<String> = tokens.iter().map(|t| t.term.clone()).collect();
                let folds = if crate::normalize::uses_raw_passthrough(language.code()) {
                    // Same start offset = same source word in two folds:
                    // one union per offset, one fold per token, so the
                    // best fold wins instead of summing. Distinct offsets
                    // (CJK bigrams et al.) and positionless tokens stand
                    // alone and sum as before.
                    let mut unions: HashMap<usize, usize> = HashMap::new();
                    let mut counts: HashMap<usize, usize> = HashMap::new();
                    let mut next_union = 0usize;
                    tokens
                        .iter()
                        .map(|t| match t.position().map(|(s, _)| s) {
                            Some(start) => {
                                let union = *unions.entry(start).or_insert_with(|| {
                                    let id = next_union;
                                    next_union += 1;
                                    id
                                });
                                let fold = counts.entry(union).or_insert(0);
                                let mine = *fold;
                                *fold += 1;
                                Some((union, mine))
                            }
                            None => None,
                        })
                        .collect()
                } else {
                    vec![None; tokens.len()]
                };
                (terms, phrases, folds)
            } else {
                // No pipeline means no second fold: normalize the raw text here
                // so programmatic clauses (`HELLO*`) behave like parsed ones,
                // and multi-language indexes cover every member's folding
                // (`Istanbul` must meet both `istanbul` and `ıstanbul`).
                // Width folding and case mapping never touch `*` or `\`.
                // One union for the whole bypass: its folds maximize.
                let bypass = Self::bypass_terms(&language, &clause.term);
                let folds: Vec<Option<(usize, usize)>> = if bypass.len() > 1 {
                    bypass
                        .iter()
                        .enumerate()
                        .map(|(ti, _)| Some((0, ti)))
                        .collect()
                } else {
                    vec![None; bypass.len()]
                };
                (bypass, Vec::new(), folds)
            };

            // Collect the expansions for this clause, deduplicated. A wildcard
            // like `**` or an overlapping fuzzy expansion can yield the same
            // indexed term repeatedly; scoring it more than once would inflate
            // the document's score by the number of duplicates.
            let mut seen: HashSet<String> = HashSet::new();
            let mut contributions: Vec<Contribution<'_>> = Vec::new();
            // Per-(document, field, union, fold) sums, maximized per union
            // below. Terms outside any union behave exactly as ungrouped sums.
            let mut fold_scores: HashMap<(&str, &str, usize, usize), f64> = HashMap::new();
            for (ti, term) in terms.iter().enumerate() {
                if term.is_empty() {
                    continue;
                }
                let fold = folds[ti];
                // Wildcard clauses bypass the pipeline, and the index keys are
                // STEMMED terms, so the pattern needs the treatment in
                // `expand_wildcard_term` rather than a single stem. Non-wildcard
                // terms were already stemmed by the pipeline (or are exact
                // bypass lookups).
                for expanded in self.expand_clause_term(&clause.term, term, clause.edit_distance) {
                    if !seen.insert(expanded.clone()) {
                        continue;
                    }
                    if let Some((key, posting)) = self.inverted_index.get_key_value(&expanded) {
                        contributions.push(Contribution {
                            term: key.as_str(),
                            posting,
                            boost: clause.boost,
                            idf: idf(self.stats.document_count, posting.document_frequency()),
                            fold,
                        });
                    }
                }
            }

            // Documents matched by this clause, per the clause's field scope.
            let mut clause_docs: HashSet<&str> = HashSet::new();
            for contribution in &contributions {
                for field_name in &clause.fields {
                    if let Some(field_docs) = contribution.posting.fields.get(field_name) {
                        clause_docs.extend(field_docs.keys().map(String::as_str));
                    }
                }
            }

            match clause.presence {
                Presence::Prohibited => {
                    prohibited.extend(clause_docs);
                    continue;
                }
                Presence::Required => {
                    // Intersect: a document must satisfy every required clause.
                    required = Some(match required {
                        Some(acc) => acc.intersection(&clause_docs).copied().collect(),
                        None => clause_docs.clone(),
                    });
                }
                Presence::Optional => {}
            }

            for contribution in &contributions {
                for field_name in &clause.fields {
                    let Some(field_docs) = contribution.posting.fields.get(field_name) else {
                        continue;
                    };
                    for doc_ref in field_docs.keys() {
                        matched_any = true;

                        // If this term belongs to a query phrase and that phrase
                        // actually occurs in this field, the match is a real
                        // phrase hit rather than a coincidence of overlapping
                        // n-grams. Boost it.
                        let phrase_boost = self.phrase_boost_for(
                            &phrases,
                            contribution.term,
                            field_name,
                            doc_ref,
                            &mut phrase_cache,
                        );

                        let score = self.score_term(
                            contribution.posting,
                            field_name,
                            doc_ref,
                            saturating_product(contribution.boost, phrase_boost),
                            contribution.idf,
                        );
                        match contribution.fold {
                            // One intent, several folds: keep the best fold's
                            // sum per document-field, merged below.
                            Some((union, fold)) => {
                                let entry = fold_scores
                                    .entry((doc_ref.as_str(), field_name.as_str(), union, fold))
                                    .or_insert(0.0);
                                *entry = saturating_add(*entry, score);
                            }
                            None => {
                                let entry = scores.entry(doc_ref.as_str()).or_insert(0.0);
                                *entry = saturating_add(*entry, score);
                            }
                        }
                    }
                }
            }
            // Match metadata is deliberately NOT collected here: see the
            // `limit`-aware pass below, which builds it only for the documents
            // that survive ranking.
            scored_clauses.push((clause.fields.as_slice(), contributions));
            // Fold groups harmonize here: each document keeps the maximum
            // across one union's folds, and every union joins the running
            // total like any other clause contribution. Keyed by union so
            // distinct intents (CJK bigrams at different offsets, separate
            // bypass terms) still sum.
            {
                let mut maxima: HashMap<(&str, &str, usize), f64> = HashMap::new();
                for ((doc_ref, field_name, union, _), sum) in fold_scores {
                    let entry = maxima.entry((doc_ref, field_name, union)).or_insert(0.0);
                    *entry = entry.max(sum);
                }
                for ((doc_ref, _, _), best) in maxima {
                    let entry = scores.entry(doc_ref).or_insert(0.0);
                    *entry = saturating_add(*entry, best);
                }
            }
        }

        // A purely negated query ("-foo") matches every document that is not
        // prohibited, with no score signal to rank by.
        let is_negated = query.is_negated();
        if is_negated && !matched_any {
            let mut results: Vec<SearchResult> = self
                .all_doc_refs()
                .into_iter()
                .filter(|d| !prohibited.contains(*d))
                .filter(|d| satisfies_required(required.as_ref(), d))
                .map(|doc_ref| SearchResult {
                    ref_id: doc_ref.to_string(),
                    score: 0.0,
                    match_data: MatchData::default(),
                })
                .collect();
            results.sort_by(|a, b| a.ref_id.cmp(&b.ref_id));
            results.truncate(limit.unwrap_or(usize::MAX));
            return results;
        }

        let mut results: Vec<SearchResult> = scores
            .into_iter()
            .filter(|(doc_ref, _)| !prohibited.contains(doc_ref))
            .filter(|(doc_ref, _)| satisfies_required(required.as_ref(), doc_ref))
            .map(|(doc_ref, score)| SearchResult {
                match_data: MatchData::default(),
                ref_id: doc_ref.to_string(),
                score,
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.ref_id.cmp(&b.ref_id))
        });
        // Truncate before building metadata, not after. Converting match data is
        // the expensive half: a broad wildcard over a 5,000-document corpus
        // produced 475,000 match-term entries and 2.3 million positions for a
        // caller that asked for ten hits.
        results.truncate(limit.unwrap_or(usize::MAX));

        self.collect_match_data(&scored_clauses, &mut results);
        results
    }

    /// Fill in [`MatchData`] for the given results only.
    ///
    /// Runs after ranking and truncation, so the work is bounded by the number
    /// of results the caller will actually see rather than by every document the
    /// query touched. Each clause's expansions are re-walked: the posting lookups
    /// are hash hits, while the alternative — collecting `(term, field,
    /// positions)` eagerly — allocates exactly the metadata this pass exists to
    /// avoid.
    fn collect_match_data(
        &self,
        scored_clauses: &[(&[String], Vec<Contribution<'_>>)],
        results: &mut [SearchResult],
    ) {
        if results.is_empty() || scored_clauses.is_empty() {
            return;
        }
        let wanted: HashSet<&str> = results.iter().map(|r| r.ref_id.as_str()).collect();
        let mut matching: HashMap<&str, MatchData> = HashMap::with_capacity(wanted.len());
        for (clause_fields, contributions) in scored_clauses {
            for contribution in contributions {
                // Only the fields the clause was scoped to, exactly like the
                // scoring pass: a `title:` clause must not report body matches.
                for field_name in clause_fields.iter() {
                    let Some(field_docs) = contribution.posting.fields.get(field_name) else {
                        continue;
                    };
                    for (doc_ref, posting_doc) in field_docs {
                        if !wanted.contains(doc_ref.as_str()) {
                            continue;
                        }
                        matching.entry(doc_ref.as_str()).or_default().add_term(
                            contribution.term,
                            field_name,
                            &posting_doc.positions,
                        );
                    }
                }
            }
        }
        for result in results.iter_mut() {
            if let Some(data) = matching.remove(result.ref_id.as_str()) {
                result.match_data = data;
            }
        }
    }

    /// Every document reference known to the index.
    fn all_doc_refs(&self) -> Vec<&str> {
        let mut refs: HashSet<&str> = HashSet::new();
        for doc_ref in self.stats.field_lengths.keys() {
            refs.insert(doc_ref.as_str());
        }
        for doc_ref in self.stats.doc_boosts.keys() {
            refs.insert(doc_ref.as_str());
        }
        refs.into_iter().collect()
    }

    /// Serialize the index to the compact binary format.
    ///
    /// Roughly a fifth of the equivalent lunr-style JSON. Pass
    /// `include_positions = false` to drop highlighting and CJK phrase
    /// verification data for a further saving of about a tenth.
    ///
    /// Fails with [`WriteError::TooLarge`](crate::binary::WriteError) when a
    /// count or offset does not fit the `u32`-addressed format — a huge index,
    /// never a small one.
    pub fn to_binary(&self, include_positions: bool) -> Result<Vec<u8>, crate::binary::WriteError> {
        crate::binary::writer::write_index(&crate::binary::writer::IndexSnapshot {
            language: self.pipeline.language().code(),
            fields: &self.fields,
            field_boosts: &self.stats.field_boosts,
            pipeline: self.pipeline.labels(),
            document_count: self.stats.document_count,
            doc_boosts: &self.stats.doc_boosts,
            field_lengths: &self.stats.field_lengths,
            inverted_index: &self.inverted_index,
            k1: self.stats.k1,
            b: self.stats.b,
            include_positions,
        })
    }

    /// Load an index from the binary format.
    ///
    /// This materializes the postings into owned in-memory structures, so
    /// search behaves the same as on a freshly built index. It is the
    /// convenient path, not the zero-copy one — use [`crate::BinaryIndex`]
    /// directly to read postings straight out of a mapped buffer.
    ///
    /// The `language` must match the one the index was built with; a mismatch
    /// makes query tokenization disagree with the indexed terms and is rejected.
    /// The recorded analysis pipeline must agree too: same code, different
    /// stemming (e.g. a trimmed build) fails the same way.
    pub fn from_binary(
        bytes: &[u8],
        language: LanguageRef,
    ) -> Result<Self, crate::binary::FormatError> {
        let binary = crate::binary::BinaryIndex::open(bytes)?;

        // Whole-structure validation before anything is allocated. `open` is
        // constant-time by contract, so the invariants only a walk can establish
        // — contiguous position blocks, sorted unique references, consistent
        // tables — are checked here, where the cost is bounded by work this call
        // is about to do anyway.
        binary.validate()?;

        // Canonical member lists, not raw strings: `"en, ja"` and `"en,ja"`
        // are the same configuration, while `"ja,en"` is genuinely different
        // (member order affects stemming).
        if crate::languages::canonical_parts(binary.language())
            != crate::languages::canonical_parts(language.code())
        {
            return Err(crate::binary::FormatError::LanguageMismatch {
                expected: language.code().to_string(),
                found: binary.language().to_string(),
            });
        }

        let expected_pipeline: Vec<String> = language
            .pipeline_labels()
            .iter()
            .map(|s| s.to_string())
            .collect();
        if binary.pipeline() != expected_pipeline.as_slice() {
            return Err(crate::binary::FormatError::PipelineMismatch {
                expected: expected_pipeline,
                found: binary.pipeline().to_vec(),
            });
        }

        // Resolve ids to strings once. A posting list references a document by
        // id many times over, so decoding the reference per reference would make
        // loading quadratic in the worst case.
        let doc_refs: Vec<String> = (0..binary.doc_count() as u32)
            .map(|id| binary.doc_ref(id).map(str::to_string))
            .collect::<Result<_, _>>()?;
        let fields = binary.fields().to_vec();
        let has_positions = binary.header().has_positions();

        let terms = binary.terms()?;
        let mut inverted_index: BTreeMap<String, Posting> = BTreeMap::new();
        for (term_id, term) in terms.into_iter().enumerate() {
            let decoded = binary.postings(term_id as u32)?;
            let mut posting = Posting::default();
            for field_postings in &decoded.fields {
                let field_name = binary.field_name(field_postings.field_id)?.to_string();
                let mut docs: HashMap<String, PostingDoc> =
                    HashMap::with_capacity(field_postings.entries.len());
                for entry in &field_postings.entries {
                    let doc_ref = doc_refs
                        .get(entry.doc_id as usize)
                        .ok_or(crate::binary::FormatError::InvalidDocId(entry.doc_id))?;
                    let positions = if has_positions {
                        binary.positions(entry)?
                    } else {
                        Vec::new()
                    };
                    docs.insert(
                        doc_ref.clone(),
                        PostingDoc {
                            term_frequency: entry.term_frequency,
                            positions,
                        },
                    );
                }
                posting.fields.insert(field_name, docs);
            }
            inverted_index.insert(term, posting);
        }

        let mut field_lengths: HashMap<String, HashMap<String, usize>> = HashMap::new();
        for (doc_id, doc_ref) in doc_refs.iter().enumerate() {
            for (field_id, field_name) in fields.iter().enumerate() {
                let length = binary.field_length(doc_id as u32, field_id as u32)?;
                field_lengths
                    .entry(doc_ref.clone())
                    .or_default()
                    .insert(field_name.clone(), length as usize);
            }
        }

        // Boosts are re-validated, not trusted: the writer clamps, so a
        // negative or non-finite value here means corrupt bytes. Loading it
        // verbatim would mint negative scores.
        let mut field_boosts = HashMap::with_capacity(fields.len());
        for (field_id, field_name) in fields.iter().enumerate() {
            let boost = binary.field_boost(field_id as u32)?;
            field_boosts.insert(field_name.clone(), sanitize_boost(boost));
        }

        let mut doc_boosts = HashMap::with_capacity(doc_refs.len());
        for (doc_id, doc_ref) in doc_refs.iter().enumerate() {
            let boost = binary.doc_boost(doc_id as u32)?;
            doc_boosts.insert(doc_ref.clone(), sanitize_boost(boost));
        }

        let stats = Stats {
            document_count: binary.document_count(),
            average_field_lengths: average_field_lengths(&field_lengths, &fields),
            field_lengths,
            field_boosts,
            doc_boosts,
            k1: sanitize_k1(binary.k1()),
            b: sanitize_b(binary.b()),
        };

        let token_set = TokenSet::from_strs(inverted_index.keys().map(String::as_str));

        Ok(Index {
            inverted_index,
            token_set,
            fields,
            stats,
            pipeline: Pipeline::new(language),
        })
    }

    /// Return the indexed field names.
    pub fn fields(&self) -> &[String] {
        &self.fields
    }

    /// Return the number of indexed documents.
    pub fn document_count(&self) -> usize {
        self.stats.document_count
    }

    /// Return the number of distinct indexed terms.
    pub fn term_count(&self) -> usize {
        self.inverted_index.len()
    }

    /// Return the language this index was built with.
    pub fn language(&self) -> LanguageRef {
        self.pipeline.language()
    }

    /// Return the BM25 `k1` parameter.
    pub fn k1(&self) -> f64 {
        self.stats.k1
    }

    /// Return the BM25 `b` parameter.
    pub fn b(&self) -> f64 {
        self.stats.b
    }

    /// Return the boost configured for `field` (`1.0` when unknown).
    pub fn field_boost(&self, field: &str) -> f64 {
        self.stats.field_boosts.get(field).copied().unwrap_or(1.0)
    }

    /// Whether a document reference is present in the index.
    pub fn contains(&self, doc_ref: &str) -> bool {
        self.stats.doc_boosts.contains_key(doc_ref)
            || self.stats.field_lengths.contains_key(doc_ref)
    }
}

/// Whether a document satisfies the accumulated set of required clauses.
///
/// `None` means the query had no required clauses, so every document passes.
///
/// Written out rather than using `Option::is_none_or`, which needs Rust 1.82
/// and would raise the crate's 1.78 MSRV.
fn satisfies_required(required: Option<&HashSet<&str>>, doc_ref: &str) -> bool {
    match required {
        Some(set) => set.contains(doc_ref),
        None => true,
    }
}

/// Clamp a boost read from untrusted bytes to the builder's contract.
///
/// Mirrors `IndexBuilder`'s normalization so a forged negative or NaN boost
/// cannot mint negative scores at query time.
fn sanitize_boost(boost: f64) -> f64 {
    if boost.is_finite() {
        boost.max(0.0)
    } else {
        1.0
    }
}

/// Clamp `k1` read from untrusted bytes to the builder's contract.
///
/// Mirrors `IndexBuilder::k1`: non-finite or negative values keep `1.2`.
fn sanitize_k1(k1: f64) -> f64 {
    if k1.is_finite() && k1 >= 0.0 {
        k1
    } else {
        1.2
    }
}

/// Clamp `b` read from untrusted bytes to the builder's contract.
///
/// Mirrors `IndexBuilder::b`: non-finite values keep `0.75`, finite ones
/// clamp to `[0, 1]`.
fn sanitize_b(b: f64) -> f64 {
    if b.is_finite() {
        b.clamp(0.0, 1.0)
    } else {
        0.75
    }
}

/// Multiply two non-negative score factors without ever producing `inf`/`NaN`.
///
/// The BM25 inputs are guarded (`bm25_weight` rejects anything non-finite), but
/// nothing guarded what happens *after*: a clause boost of `1e308` times a
/// phrase boost of 100 is `inf`, and an infinite boost times a zero weight is
/// `NaN`. Both used to reach the caller — one document can be searched with a
/// long repeated CJK phrase, or with eight `term^1.7e308` clauses.
///
/// One policy for every multiplication and sum in scoring: zero anywhere means
/// zero (short-circuited, so `0 × inf` cannot become `NaN`), and overflow
/// saturates at [`f64::MAX`] rather than extrapolating. Saturating keeps
/// ranking meaningful — the overflowing document still sorts first — while
/// guaranteeing the number JavaScript receives is finite.
///
/// Complementary to `MAX_CLAUSE_BOOST` (which bounds one clause-boost input) and
/// to `sanitize_k1`/`sanitize_b` (which bound the BM25 parameters on load): those
/// keep the operands sane, this keeps the arithmetic finite. The phrase boost is
/// `PHRASE_BOOST` per verified phrase, so a long repeated CJK phrase overflows
/// with no absurd input at all.
fn saturating_product(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        return 0.0;
    }
    let product = a * b;
    if product.is_finite() {
        product
    } else if product.is_nan() {
        // Only reachable with mixed signs, which the guards above exclude; a
        // score that cannot be computed contributes nothing.
        0.0
    } else {
        f64::MAX
    }
}

/// Add two score contributions without ever producing `inf`/`NaN`.
///
/// See [`saturating_product`]: eight finite-but-huge clause scores used to sum
/// to `inf`, which sorts unpredictably once it reaches the caller.
fn saturating_add(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        return if b.is_finite() { b } else { 0.0 };
    }
    if b == 0.0 {
        return a;
    }
    let sum = a + b;
    if sum.is_finite() {
        sum
    } else if sum.is_nan() {
        0.0
    } else {
        f64::MAX
    }
}

/// Recompute mean field lengths from the per-field-ref lengths.
///
/// The averages are derived on load rather than serialized, since they are a
/// pure function of data already present and would otherwise be one more thing
/// that can disagree with itself. Empty fields are excluded: the average is
/// over docs that *have* the field.
fn average_field_lengths(
    field_lengths: &HashMap<String, HashMap<String, usize>>,
    fields: &[String],
) -> HashMap<String, f64> {
    let mut total: HashMap<&str, usize> = HashMap::new();
    let mut count: HashMap<&str, usize> = HashMap::new();
    for fields_map in field_lengths.values() {
        for (field_name, len) in fields_map {
            if *len == 0 {
                continue;
            }
            *total.entry(field_name.as_str()).or_insert(0) += len;
            *count.entry(field_name.as_str()).or_insert(0) += 1;
        }
    }
    fields
        .iter()
        .map(|name| {
            let sum = total.get(name.as_str()).copied().unwrap_or(0) as f64;
            let n = count.get(name.as_str()).copied().unwrap_or(0) as f64;
            let avg = if n > 0.0 { sum / n } else { 0.0 };
            (name.clone(), avg)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::languages::English;
    use std::sync::Arc;

    fn en() -> LanguageRef {
        Arc::new(English)
    }

    fn test_index() -> Index {
        let mut builder = IndexBuilder::new(en());
        builder
            .ref_field("id")
            .field("title", 1.0)
            .field("body", 1.0);
        builder.add("a", 1.0, |name| match name {
            "title" => Some("Mr. Green kills Colonel Mustard".to_string()),
            "body" => Some(
                "Mr. Green killed Colonel Mustard in the study with the candlestick.".to_string(),
            ),
            _ => None,
        });
        builder.add("b", 1.0, |name| match name {
            "title" => Some("Plumb water green plants".to_string()),
            "body" => Some("Professor Plumb has a green plant in his study".to_string()),
            _ => None,
        });
        builder.build()
    }

    #[test]
    fn index_has_terms_and_stats() {
        let index = test_index();
        assert!(!index.inverted_index.is_empty());
        assert_eq!(index.document_count(), 2);
        assert!(!index.stats.average_field_lengths.is_empty());
    }

    #[test]
    fn search_finds_matching_documents() {
        let index = test_index();
        let results = index.search("green").unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|r| r.ref_id == "a"));
        assert!(results.iter().any(|r| r.ref_id == "b"));
    }

    #[test]
    fn search_ranks_documents_by_relevance() {
        let index = test_index();
        let results = index.search("study").unwrap();
        assert_eq!(results[0].ref_id, "b");
    }

    #[test]
    fn search_required_term() {
        let index = test_index();
        let results = index.search("+green +candlestick").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].ref_id, "a");
    }

    #[test]
    fn search_prohibited_term() {
        let index = test_index();
        let results = index.search("green -plumb").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].ref_id, "a");
    }

    #[test]
    fn search_field_scoped() {
        let index = test_index();
        let results = index.search("title:green").unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_wildcard() {
        let index = test_index();
        let results = index.search("pl*").unwrap();
        assert!(results.iter().any(|r| r.ref_id == "b"));
    }

    #[test]
    fn search_wildcard_matches_surface_form_of_stemmed_terms() {
        // The token set holds STEMMED terms (`searchable` indexes as
        // `searchabl`). A wildcard clause bypasses the pipeline, so expansion
        // of the raw surface form `searchable*` must still reach `searchabl`:
        // the pattern body is stemmed first, with the star re-attached.
        // Without this, every word whose stem differs from its surface form
        // silently returned zero hits — the common typeahead case.
        let mut builder = IndexBuilder::new(en());
        builder.ref_field("id").field("body", 1.0);
        builder.add("d", 1.0, |_| {
            Some("searchable readable content".to_string())
        });
        let index = builder.build();
        for query in ["searchable*", "readable*"] {
            let hits = index.search(query).unwrap();
            assert_eq!(
                hits.iter().map(|h| h.ref_id.as_str()).collect::<Vec<_>>(),
                vec!["d"],
                "surface form + star must meet its stem: {query}"
            );
        }
        // Stem form + star still works, and bare terms keep matching.
        for query in ["searchabl*", "searchable"] {
            assert!(!index.search(query).unwrap().is_empty(), "{query}");
        }
        // Prefix of the stem keeps its wildcard recall (lunr typeahead).
        assert!(!index.search("search*").unwrap().is_empty());
    }

    #[test]
    fn search_leading_wildcard_with_stemmed_body() {
        // Leading star + body: the star survives round-tripping the stem.
        let mut builder = IndexBuilder::new(en());
        builder.ref_field("id").field("body", 1.0);
        builder.add("d", 1.0, |_| Some("candlestick besmoothed".to_string()));
        let index = builder.build();
        // stem("andlestick") == "andleystick"; "*andlestick*" must still
        // find the indexed "candlestick" (stem "candlestick").
        let hits = index.search("*andlestick*").unwrap();
        assert!(hits.iter().any(|h| h.ref_id == "d"));
    }

    #[test]
    fn search_fuzzy() {
        let index = test_index();
        let results = index.search("stud~1").unwrap();
        assert!(results.iter().any(|r| r.ref_id == "a" || r.ref_id == "b"));
    }

    #[test]
    fn fuzzy_search_finds_a_transposed_typo() {
        // End to end through the query parser and the pipeline: the swap must
        // survive stemming and still cost only one edit. Under plain
        // Levenshtein this query would return nothing.
        let mut builder = IndexBuilder::new(en());
        builder.ref_field("id").field("body", 1.0);
        builder.add("a", 1.0, |_| Some("keyboard".to_string()));
        let index = builder.build();

        assert!(
            !index.search("keybaord~1").unwrap().is_empty(),
            "a transposed pair should be one edit away"
        );
        assert!(
            index.search("keybaord").unwrap().is_empty(),
            "without ~1 the typo should still not match"
        );
    }

    #[test]
    fn wildcard_does_not_inflate_scores_with_duplicates() {
        // `**` expands to every term, and the trie can reach the same term by
        // several paths. Each indexed term must still be counted once.
        let index = test_index();
        let star = index.search("**").unwrap();
        let all: Vec<_> = index
            .inverted_index
            .keys()
            .map(|t| index.search(t).unwrap())
            .collect();
        assert!(!star.is_empty());
        // Every document should appear exactly once in the result list.
        let mut refs: Vec<&str> = star.iter().map(|r| r.ref_id.as_str()).collect();
        let before = refs.len();
        refs.sort_unstable();
        refs.dedup();
        assert_eq!(before, refs.len(), "duplicate documents in results");
        assert!(!all.is_empty());
    }

    #[test]
    fn match_data_positions_are_deduplicated() {
        let index = test_index();
        let results = index.search("green gre*").unwrap();
        for result in &results {
            for fields in result.match_data.terms.values() {
                for positions in fields.values() {
                    let mut sorted = positions.clone();
                    let before = sorted.len();
                    sorted.sort_unstable();
                    sorted.dedup();
                    assert_eq!(before, sorted.len(), "duplicate positions in match data");
                }
            }
        }
    }

    #[test]
    fn field_ref_roundtrips_through_string() {
        // A docref containing slashes is normal: they are usually URL paths.
        let field_ref = FieldRef::new("guide/install/index.html", "title");
        let parsed = FieldRef::from_string(&field_ref.to_string()).unwrap();
        assert_eq!(parsed, field_ref);
        assert_eq!(parsed.doc_ref(), "guide/install/index.html");
        assert_eq!(parsed.field_name(), "title");
    }

    #[test]
    fn field_ref_rejects_input_without_separator() {
        assert!(FieldRef::from_string("nofieldsep").is_none());
        // Must not panic on multi-byte input.
        assert!(FieldRef::from_string("é").is_none());
    }

    #[test]
    fn zero_boost_suppresses_a_clause() {
        let index = test_index();
        let scored = index.search("green").unwrap();
        let zeroed = index.search("green^0").unwrap();
        assert!(scored.iter().all(|r| r.score > 0.0));
        assert!(
            zeroed.iter().all(|r| r.score == 0.0),
            "a ^0 boost must contribute no score"
        );
    }

    #[test]
    fn huge_boost_scores_finite() {
        // `^1e308` is finite, so the parser accepts it — but an unclamped
        // clause boost overflows every score it touches to infinity, which
        // sorts arbitrarily. The clamp keeps scores finite.
        let index = test_index();
        let results = index.search("green^1e308").unwrap();
        assert!(!results.is_empty());
        assert!(
            results.iter().all(|r| r.score.is_finite()),
            "scores must stay finite: {:?}",
            results.iter().map(|r| r.score).collect::<Vec<_>>()
        );
    }

    #[test]
    fn from_binary_sanitizes_k1_and_b() {
        // Forged BM25 parameters must load as the builder's clamps, not
        // verbatim: NaN `k1` or an out-of-range `b` would corrupt scoring.
        // `k1` lives at header bytes 24..32, `b` at 32..40.
        let bytes = test_index().to_binary(true).unwrap();
        let mut bad_k1 = bytes.clone();
        bad_k1[24..32].copy_from_slice(&f64::NAN.to_le_bytes());
        assert_eq!(Index::from_binary(&bad_k1, en()).unwrap().k1(), 1.2);
        let mut bad_b = bytes.clone();
        bad_b[32..40].copy_from_slice(&5.0f64.to_le_bytes());
        assert_eq!(Index::from_binary(&bad_b, en()).unwrap().b(), 1.0);
    }

    #[test]
    fn duplicate_doc_ref_replaces_instead_of_merging() {
        // Upsert: re-adding a ref drops its old postings and leaves N alone.
        // Merging would double `tf` while overwriting field lengths.
        let mut builder = IndexBuilder::new(en());
        builder.ref_field("id").field("body", 1.0);
        builder.add("d", 1.0, |_| Some("green green green".to_string()));
        builder.add("d", 1.0, |_| Some("plumb".to_string()));
        let index = builder.build();
        assert_eq!(index.document_count(), 1);
        assert!(index.search("green").unwrap().is_empty());
        assert_eq!(index.search("plumb").unwrap().len(), 1);
    }

    #[test]
    fn from_index_patches_without_full_rebuild() {
        // Incremental path: serialize, reload, rehydrate a builder, remove one
        // document and upsert another, then rebuild. The on-disk bytes are
        // never patched in place — the builder re-indexes only what changed.
        let mut builder = IndexBuilder::new(en());
        builder
            .ref_field("id")
            .field("title", 1.0)
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
        let bytes = builder.build().to_binary(true).unwrap();

        let loaded = Index::from_binary(&bytes, en()).expect("roundtrip fixture");
        assert_eq!(loaded.document_count(), 2);
        let mut patched = IndexBuilder::from_index(loaded);
        assert!(patched.remove("a"));
        assert!(!patched.remove("missing"));
        assert!(!patched.remove("a"));
        patched.add("c", 1.0, |name| match name {
            "title" => Some("green keyboard".to_string()),
            "body" => Some("typing".to_string()),
            _ => None,
        });
        // Upsert through the rehydrated builder replaces without growing N.
        patched.add("b", 1.0, |name| match name {
            "title" => Some("brass keyboard".to_string()),
            "body" => Some("typing".to_string()),
            _ => None,
        });
        let index = patched.build();
        assert_eq!(index.document_count(), 2);
        assert!(index.search("orchard").unwrap().is_empty());
        assert!(index.search("plumb").unwrap().is_empty());
        let hits = index.search("keyboard").unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().any(|h| h.ref_id == "b"));
        assert!(hits.iter().any(|h| h.ref_id == "c"));
    }

    #[test]
    fn turkish_query_matches_turkish_indexing() {
        use crate::languages::registry;
        let tr = registry::resolve("tr").language;
        let mut builder = IndexBuilder::new(tr);
        builder.ref_field("id").field("body", 1.0);
        builder.add("d", 1.0, |_| Some("Istanbul".to_string()));
        let index = builder.build();
        for query in ["Istanbul", "ISTANBUL", "ıstanbul"] {
            assert_eq!(
                index.search(query).unwrap().len(),
                1,
                "query {query:?} must meet indexed ıstanbul"
            );
        }
    }

    #[test]
    fn escaped_star_is_literal_not_wildcard() {
        // `a\*b` must find only the literal term, while `a*` wildcard-matches
        // everything starting with `a` — the two must not score identically.
        // Korean has no stemmer, so the literal term survives indexing intact.
        use crate::languages::Korean;
        let ko: LanguageRef = std::sync::Arc::new(Korean);
        let mut builder = IndexBuilder::new(ko);
        builder.ref_field("id").field("body", 1.0);
        builder.add("star", 1.0, |_| Some("a*b test".to_string()));
        builder.add("other", 1.0, |_| Some("axb test".to_string()));
        let index = builder.build();
        let literal = index.search(r"a\*b").unwrap();
        assert_eq!(
            literal.len(),
            1,
            "escaped star must match only the literal term"
        );
        assert_eq!(literal[0].ref_id, "star");
        let wild = index.search("a*").unwrap();
        assert!(
            wild.len() > literal.len(),
            "wildcard must match more than the literal"
        );
    }

    #[test]
    #[cfg(feature = "de")]
    fn from_binary_rejects_a_different_pipeline() {
        // Same code, different stemming: full Snowball German vs the generic
        // fallback a trimmed build resolves it to. Must fail loudly — the
        // alternative is empty results with no error.
        use crate::languages::Generic;

        let stemmed = crate::languages::registry::resolve("de").language;
        let mut builder = IndexBuilder::new(stemmed);
        builder.ref_field("id").field("body", 1.0);
        builder.add("d", 1.0, |_| Some("suchmaschinen".to_string()));
        let bytes = builder.build().to_binary(true).unwrap();

        let unstemmed: LanguageRef = std::sync::Arc::new(Generic::new("de"));
        assert!(matches!(
            Index::from_binary(&bytes, unstemmed),
            Err(crate::binary::FormatError::PipelineMismatch { .. })
        ));
    }

    #[test]
    fn escaped_star_with_fuzzy_is_exact_on_the_literal() {
        // `~0` must behave like the exact lookup: `a\*b~0` finds indexed
        // `a*b` and nothing else. With `~N>0` the `*` is an ordinary matrix
        // character, so one substitution still reaches `axb` — fuzzy, not
        // exact, and documented as such in the query syntax.
        use crate::languages::Korean;
        let ko: LanguageRef = std::sync::Arc::new(Korean);
        let mut builder = IndexBuilder::new(ko);
        builder.ref_field("id").field("body", 1.0);
        builder.add("star", 1.0, |_| Some("a*b test".to_string()));
        builder.add("other", 1.0, |_| Some("axb test".to_string()));
        let index = builder.build();
        let exact = index.search(r"a\*b~0").unwrap();
        assert_eq!(
            exact.len(),
            1,
            "~0 on a literal must be exclusive, got {:?}",
            exact.iter().map(|h| &h.ref_id).collect::<Vec<_>>()
        );
        assert_eq!(exact[0].ref_id, "star");
        let hits = index.search(r"a\*b~1").unwrap();
        assert!(
            hits.iter().any(|h| h.ref_id == "star"),
            "fuzzy escaped star must find the literal term"
        );
    }

    #[test]
    fn mixed_escaped_and_bare_stars_combine() {
        // Literal `a*b` prefix plus a trailing wildcard: matches `a*b...`
        // terms, not every `a…` term.
        use crate::languages::Korean;
        let ko: LanguageRef = std::sync::Arc::new(Korean);
        let mut builder = IndexBuilder::new(ko);
        builder.ref_field("id").field("body", 1.0);
        builder.add("star", 1.0, |_| Some("a*b test".to_string()));
        builder.add("other", 1.0, |_| Some("axb test".to_string()));
        let index = builder.build();
        let hits = index.search(r"a\*b*").unwrap();
        assert_eq!(
            hits.len(),
            1,
            "literal prefix plus wildcard must match only a*b terms"
        );
        assert_eq!(hits[0].ref_id, "star");
    }

    #[test]
    fn stale_wildcard_flag_does_not_disable_stemming() {
        // A programmatically set `has_wildcard` with no `*` in the term is
        // cleared: `running` must still stem to indexed `run`.
        use crate::query::{Clause, Query};
        let mut builder = IndexBuilder::new(en());
        builder.ref_field("id").field("body", 1.0);
        builder.add("a", 1.0, |_| Some("running running".to_string()));
        let index = builder.build();
        let mut query = Query::new(vec!["body".to_string()]);
        query.clause(Clause {
            term: "running".to_string(),
            has_wildcard: true,
            ..Clause::default()
        });
        let hits = index.query(&query);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].ref_id, "a");
    }

    #[test]
    fn programmatic_wildcard_folds_like_parsed() {
        // `Query::term("HELLO*")` must behave exactly like `search("HELLO*"):
        // the bypass path folds raw terms instead of trusting their case.
        let mut builder = IndexBuilder::new(en());
        builder.ref_field("id").field("body", 1.0);
        builder.add("a", 1.0, |_| Some("hello world".to_string()));
        let index = builder.build();
        assert_eq!(index.search("HELLO*").unwrap().len(), 1);
        let mut query = Query::new(vec!["body".to_string()]);
        query.term("HELLO*");
        assert_eq!(index.query(&query).len(), 1);
    }

    #[test]
    fn multi_turkish_wildcard_covers_both_folds() {
        // `ISTANBUL*` must meet `ıstanbul` (Turkish fold); `HELLO*` meets
        // `hello` (default fold). One bypass, two folds, unioned.
        use crate::languages::registry;
        let multi = registry::resolve_multi("en,tr").language;
        let mut builder = IndexBuilder::new(multi);
        builder.ref_field("id").field("body", 1.0);
        builder.add("en-doc", 1.0, |_| Some("hello world".to_string()));
        builder.add("tr-doc", 1.0, |_| Some("Istanbul".to_string()));
        let index = builder.build();
        assert_eq!(
            index
                .search("HELLO*")
                .unwrap()
                .iter()
                .map(|h| &h.ref_id)
                .collect::<Vec<_>>(),
            ["en-doc"]
        );
        assert_eq!(
            index
                .search("ISTANBUL*")
                .unwrap()
                .iter()
                .map(|h| &h.ref_id)
                .collect::<Vec<_>>(),
            ["tr-doc"]
        );
    }

    #[test]
    fn empty_term_matches_nothing() {
        use crate::query::{Clause, Query};
        let index = test_index();
        let mut query = Query::new(vec!["title".to_string(), "body".to_string()]);
        query.clause(Clause {
            term: String::new(),
            wildcard: crate::query::Wildcard::Trailing,
            ..Clause::default()
        });
        assert!(index.query(&query).is_empty());
    }

    #[test]
    #[should_panic(expected = "field name must not be empty")]
    fn empty_field_name_panics() {
        IndexBuilder::new(en()).field("", 1.0);
    }

    #[test]
    #[should_panic(expected = "must not contain '/'")]
    fn slash_field_name_panics() {
        IndexBuilder::new(en()).field("a/b", 1.0);
    }

    #[test]
    #[should_panic(expected = "query operator")]
    fn operator_field_name_panics() {
        IndexBuilder::new(en()).field("a:b", 1.0);
    }

    #[test]
    fn fold_union_scores_once_not_twice() {
        // Same intent in two folds must not double-count: `ISTANBUL*` meets
        // both the `istanbul` and the `ıstanbul` posting of one word, but the
        // document scores the maximum, not the sum — equal to the lowercase
        // query that needs only one fold.
        use crate::languages::registry;
        let multi = registry::resolve_multi("en,tr").language;
        let mut builder = IndexBuilder::new(multi);
        builder.ref_field("id").field("body", 1.0);
        builder.add("d", 1.0, |_| Some("Istanbul".to_string()));
        let index = builder.build();
        let upper = index.search("ISTANBUL*").unwrap()[0].score;
        let lower = index.search("istanbul*").unwrap()[0].score;
        assert!(
            (upper - lower).abs() < 1e-9,
            "fold union must not double-count: {upper} vs {lower}"
        );
    }

    #[test]
    fn pipeline_folds_max_while_terms_sum() {
        // Parsed non-wildcard queries in passthrough mode: two folds of one
        // word take the maximum, while two different words still sum.
        use crate::languages::registry;
        let multi = registry::resolve_multi("en,tr").language;
        let mut builder = IndexBuilder::new(multi);
        builder.ref_field("id").field("body", 1.0);
        builder.add("d", 1.0, |_| Some("Istanbul Ankara".to_string()));
        let index = builder.build();
        let both = index.search("ISTANBUL ANKARA").unwrap()[0].score;
        let one = index.search("ISTANBUL").unwrap()[0].score;
        assert!(
            both > one,
            "distinct query words must still sum: {both} vs {one}"
        );
        let upper = index.search("ISTANBUL").unwrap()[0].score;
        let lower = index.search("istanbul").unwrap()[0].score;
        assert!(
            (upper - lower).abs() < 1e-9,
            "same intent must score equally: {upper} vs {lower}"
        );
    }
}
