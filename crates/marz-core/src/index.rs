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
//! * **Correctness under boosts.** A precomputed weight bakes in the field
//!   boost and document boost that were configured at build time, so the same
//!   index cannot be re-queried with different boosts.
//! * **Zero-copy.** A sparse `f64` vector per document-field cannot be read
//!   from a memory-mapped byte slice without allocating. Term frequencies and
//!   field lengths can.
//!
//! Only three things are needed at query time to compute BM25: the term's
//! document frequency (the posting list length), the term frequency in the
//! field (stored per posting), and the field length plus its corpus average
//! (stored per document-field). All three are small integers.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::language::LanguageRef;
use crate::phrase::{extract_phrases, Phrase, VerificationCache, PHRASE_BOOST};
use crate::pipeline::Pipeline;
use crate::query::{Presence, Query};
use crate::query_parser::{parse_query, QueryParseError};
use crate::token_set::TokenSet;
use crate::{bm25_weight, idf};

/// Marz index serialization format version.
///
/// This is Marz's own format version and is unrelated to any lunr version.
/// Marz indexes are not interchangeable with lunr indexes: the CJK
/// tokenization differs, and the scoring data is stored as term frequencies
/// rather than precomputed field vectors.
const INDEX_VERSION: &str = "3";

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
        let entry = self
            .terms
            .entry(term.to_string())
            .or_default()
            .entry(field.to_string())
            .or_default();
        for position in positions {
            if !entry.contains(position) {
                entry.push(*position);
            }
        }
    }

    /// Merge another match data object into this one.
    pub fn merge(&mut self, other: &Self) {
        for (term, fields) in &other.terms {
            let entry = self.terms.entry(term.clone()).or_default();
            for (field, positions) in fields {
                let target = entry.entry(field.clone()).or_default();
                for position in positions {
                    if !target.contains(position) {
                        target.push(*position);
                    }
                }
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
    /// Length in tokens of each document field.
    field_lengths: HashMap<FieldRef, usize>,
    /// Mean field length per field name, over documents that have the field.
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
pub struct IndexBuilder {
    language: LanguageRef,
    ref_field: String,
    fields: Vec<FieldConfig>,
    field_lengths: HashMap<FieldRef, usize>,
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
        self.ref_field = field.into();
        self
    }

    /// Return the configured document reference field.
    pub fn ref_field_name(&self) -> &str {
        &self.ref_field
    }

    /// Add a field to the index. `boost` defaults to `1.0`.
    pub fn field(&mut self, name: impl Into<String>, boost: f64) -> &mut Self {
        self.fields.push(FieldConfig {
            name: name.into(),
            boost: boost.max(0.0),
        });
        self
    }

    /// Set the BM25 `k1` parameter.
    pub fn k1(&mut self, k1: f64) -> &mut Self {
        self.k1 = k1;
        self
    }

    /// Set the BM25 `b` parameter, clamped to `[0, 1]`.
    pub fn b(&mut self, b: f64) -> &mut Self {
        self.b = b.clamp(0.0, 1.0);
        self
    }

    /// Add a document to the index.
    ///
    /// `field_getter` receives a field name and should return the raw text for
    /// that field, or `None` if the field is missing.
    pub fn add<F>(&mut self, doc_ref: impl Into<String>, doc_boost: f64, mut field_getter: F)
    where
        F: FnMut(&str) -> Option<String>,
    {
        let doc_ref = doc_ref.into();
        self.doc_boosts.insert(doc_ref.clone(), doc_boost.max(0.0));
        self.document_count += 1;

        let pipeline = Pipeline::new(self.language.clone());

        for field in &self.fields {
            let text = field_getter(&field.name).unwrap_or_default();
            let tokens = self.language.tokenize(&text);
            let terms = pipeline.run_index(tokens);

            let field_ref = FieldRef::new(doc_ref.clone(), field.name.clone());
            self.field_lengths.insert(field_ref, terms.len());

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

    /// Consume the builder and produce a searchable [`Index`].
    pub fn build(self) -> Index {
        let average_field_lengths = self.calculate_average_field_lengths();
        let token_set =
            TokenSet::from_sorted(&self.inverted_index.keys().cloned().collect::<Vec<_>>());

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

    fn calculate_average_field_lengths(&self) -> HashMap<String, f64> {
        let mut total: HashMap<&str, usize> = HashMap::new();
        let mut count: HashMap<&str, usize> = HashMap::new();

        for (field_ref, length) in &self.field_lengths {
            *total.entry(field_ref.field_name.as_str()).or_insert(0) += length;
            *count.entry(field_ref.field_name.as_str()).or_insert(0) += 1;
        }

        self.fields
            .iter()
            .map(|field| {
                let sum = total.get(field.name.as_str()).copied().unwrap_or(0) as f64;
                let n = count.get(field.name.as_str()).copied().unwrap_or(1).max(1) as f64;
                (field.name.clone(), sum / n)
            })
            .collect()
    }
}

/// Built search index.
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
}

impl Index {
    /// Search the index using lunr query syntax.
    ///
    /// Supports `+` required, `-` prohibited, `field:term`, `term^boost`,
    /// `term~edits`, and `*` wildcards.
    pub fn search(&self, query_string: &str) -> Result<Vec<SearchResult>, QueryParseError> {
        let language = self.pipeline.language();
        let separators = language.separator_chars();
        let query = parse_query(query_string, &self.fields, separators)?;
        Ok(self.execute_query(&query))
    }

    /// Execute a programmatic [`Query`] against the index.
    pub fn query(&self, query: &Query) -> Vec<SearchResult> {
        self.execute_query(query)
    }

    /// BM25 score for one term occurrence in one document field.
    fn score_term(&self, posting: &Posting, field_name: &str, doc_ref: &str, boost: f64) -> f64 {
        let Some(posting_doc) = posting
            .fields
            .get(field_name)
            .and_then(|docs| docs.get(doc_ref))
        else {
            return 0.0;
        };

        let field_ref = FieldRef::new(doc_ref, field_name);
        let field_length = self
            .stats
            .field_lengths
            .get(&field_ref)
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

        let idf_value = idf(self.stats.document_count, posting.document_frequency());
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
        weight * boost
    }

    /// Expand a clause term into the set of indexed terms it matches.
    fn expand_clause_term(
        &self,
        clause_term: &str,
        term: &str,
        edits: Option<usize>,
    ) -> Vec<String> {
        if clause_term.contains('*') {
            self.token_set.expand(term)
        } else if let Some(edits) = edits {
            self.token_set.expand_fuzzy(term, edits)
        } else {
            self.token_set.expand(term)
        }
    }

    /// Boost multiplier for `term` in one document field, from phrase matches.
    ///
    /// Returns `1.0` (no change) unless `term` belongs to a query phrase that
    /// genuinely occurs in this field. Word-tokenized languages never produce
    /// phrases, so they take the empty-slice fast path and pay nothing.
    fn phrase_boost_for(
        &self,
        phrases: &[Phrase],
        term: &str,
        field_name: &str,
        doc_ref: &str,
        cache: &mut VerificationCache,
    ) -> f64 {
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
                return PHRASE_BOOST;
            }
        }
        1.0
    }

    fn execute_query(&self, query: &Query) -> Vec<SearchResult> {
        let language = self.pipeline.language();
        let mut scores: HashMap<String, f64> = HashMap::new();
        let mut matching: HashMap<String, MatchData> = HashMap::new();
        let mut required: Option<HashSet<String>> = None;
        let mut prohibited: HashSet<String> = HashSet::new();
        let mut matched_any = false;
        // Phrase verification is per (phrase, field, document) but is consulted
        // once per phrase *term*, so the verdict is cached.
        let mut phrase_cache = VerificationCache::default();

        for clause in &query.clauses {
            // Phrases are derived from the pipeline's tokens, so they are only
            // available when the pipeline ran. A wildcard clause disables it,
            // and a wildcard is an explicit request for loose matching anyway.
            let (terms, phrases) = if clause.use_pipeline {
                let tokens = self.pipeline.run_search(&clause.term);
                let phrases = extract_phrases(&tokens, &language);
                let terms: Vec<String> = tokens.into_iter().map(|t| t.term).collect();
                (terms, phrases)
            } else {
                (vec![clause.term.clone()], Vec::new())
            };

            // Collect the expansions for this clause, deduplicated. A wildcard
            // like `**` or an overlapping fuzzy expansion can yield the same
            // indexed term repeatedly; scoring it more than once would inflate
            // the document's score by the number of duplicates.
            let mut seen: HashSet<String> = HashSet::new();
            let mut contributions: Vec<Contribution<'_>> = Vec::new();
            for term in &terms {
                if term.is_empty() {
                    continue;
                }
                for expanded in self.expand_clause_term(&clause.term, term, clause.edit_distance) {
                    if !seen.insert(expanded.clone()) {
                        continue;
                    }
                    if let Some((key, posting)) = self.inverted_index.get_key_value(&expanded) {
                        contributions.push(Contribution {
                            term: key.as_str(),
                            posting,
                            boost: clause.boost,
                        });
                    }
                }
            }

            // Documents matched by this clause, per the clause's field scope.
            let mut clause_docs: HashSet<String> = HashSet::new();
            for contribution in &contributions {
                for field_name in &clause.fields {
                    if let Some(field_docs) = contribution.posting.fields.get(field_name) {
                        clause_docs.extend(field_docs.keys().cloned());
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
                        Some(acc) => acc.intersection(&clause_docs).cloned().collect(),
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
                    for (doc_ref, posting_doc) in field_docs {
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
                            contribution.boost * phrase_boost,
                        );
                        *scores.entry(doc_ref.clone()).or_insert(0.0) += score;
                        matching.entry(doc_ref.clone()).or_default().add_term(
                            contribution.term,
                            field_name,
                            &posting_doc.positions,
                        );
                    }
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
                .filter(|d| !prohibited.contains(d))
                .filter(|d| satisfies_required(required.as_ref(), d))
                .map(|doc_ref| SearchResult {
                    ref_id: doc_ref,
                    score: 0.0,
                    match_data: MatchData::default(),
                })
                .collect();
            results.sort_by(|a, b| a.ref_id.cmp(&b.ref_id));
            return results;
        }

        let mut results: Vec<SearchResult> = scores
            .into_iter()
            .filter(|(doc_ref, _)| !prohibited.contains(doc_ref))
            .filter(|(doc_ref, _)| satisfies_required(required.as_ref(), doc_ref))
            .map(|(doc_ref, score)| SearchResult {
                match_data: matching.get(&doc_ref).cloned().unwrap_or_default(),
                ref_id: doc_ref,
                score,
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.ref_id.cmp(&b.ref_id))
        });
        results
    }

    /// Every document reference known to the index.
    fn all_doc_refs(&self) -> Vec<String> {
        let mut refs: HashSet<&str> = HashSet::new();
        for field_ref in self.stats.field_lengths.keys() {
            refs.insert(field_ref.doc_ref.as_str());
        }
        refs.into_iter().map(String::from).collect()
    }

    /// Serialize the index to a JSON string.
    pub fn to_json(&self) -> String {
        let inverted_index: Vec<(String, SerializedPosting)> = self
            .inverted_index
            .iter()
            .map(|(term, posting)| {
                let fields = posting
                    .fields
                    .iter()
                    .filter(|(_, docs)| !docs.is_empty())
                    .map(|(field_name, docs)| {
                        let docs = docs
                            .iter()
                            .map(|(doc_ref, posting_doc)| {
                                (
                                    doc_ref.clone(),
                                    SerializedPostingDoc {
                                        term_frequency: posting_doc.term_frequency,
                                        positions: posting_doc.positions.clone(),
                                    },
                                )
                            })
                            .collect();
                        (field_name.clone(), docs)
                    })
                    .collect();
                (term.clone(), SerializedPosting { fields })
            })
            .collect();

        let mut field_lengths: Vec<(String, usize)> = self
            .stats
            .field_lengths
            .iter()
            .map(|(field_ref, len)| (field_ref.to_string(), *len))
            .collect();
        field_lengths.sort();

        let mut doc_boosts: Vec<(String, f64)> = self
            .stats
            .doc_boosts
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        doc_boosts.sort_by(|a, b| a.0.cmp(&b.0));

        let serialized = SerializedIndex {
            version: INDEX_VERSION.to_string(),
            language: self.pipeline.language().code().to_string(),
            fields: self.fields.clone(),
            field_boosts: self
                .fields
                .iter()
                .map(|f| self.stats.field_boosts.get(f).copied().unwrap_or(1.0))
                .collect(),
            document_count: self.stats.document_count,
            k1: self.stats.k1,
            b: self.stats.b,
            field_lengths,
            doc_boosts,
            inverted_index,
            pipeline: self.pipeline.labels(),
        };

        serde_json::to_string(&serialized).expect("index serialization")
    }

    /// Load a previously serialized index.
    ///
    /// The `language` must match the tokenizer/pipeline used to build the index.
    pub fn load(json: &str, language: LanguageRef) -> Result<Self, serde_json::Error> {
        let serialized: SerializedIndex = serde_json::from_str(json)?;

        let mut inverted_index: BTreeMap<String, Posting> = BTreeMap::new();
        for (term, sp) in serialized.inverted_index {
            let mut posting = Posting::default();
            for (field_name, docs) in sp.fields {
                let doc_map = docs
                    .into_iter()
                    .map(|(doc_ref, spd)| {
                        // Fall back to the position count when the term
                        // frequency is absent, so an index written without it
                        // still scores correctly.
                        let term_frequency = if spd.term_frequency > 0 {
                            spd.term_frequency
                        } else {
                            spd.positions.len().max(1) as u32
                        };
                        (
                            doc_ref,
                            PostingDoc {
                                term_frequency,
                                positions: spd.positions,
                            },
                        )
                    })
                    .collect();
                posting.fields.insert(field_name, doc_map);
            }
            inverted_index.insert(term, posting);
        }

        let field_lengths = serialized
            .field_lengths
            .into_iter()
            .filter_map(|(key, len)| FieldRef::from_string(&key).map(|fr| (fr, len)))
            .collect();

        let field_boosts = serialized
            .fields
            .iter()
            .cloned()
            .zip(
                serialized
                    .field_boosts
                    .iter()
                    .copied()
                    .chain(std::iter::repeat(1.0)),
            )
            .collect();

        let stats = Stats {
            document_count: serialized.document_count,
            average_field_lengths: average_field_lengths(&field_lengths, &serialized.fields),
            field_lengths,
            field_boosts,
            doc_boosts: serialized.doc_boosts.into_iter().collect(),
            k1: serialized.k1,
            b: serialized.b,
        };

        let token_set = TokenSet::from_sorted(&inverted_index.keys().cloned().collect::<Vec<_>>());

        Ok(Index {
            inverted_index,
            token_set,
            fields: serialized.fields,
            stats,
            pipeline: Pipeline::new(language),
        })
    }

    /// Serialize the index to the compact binary format.
    ///
    /// Roughly a fifth the size of [`Index::to_json`] output. Pass
    /// `include_positions = false` to drop highlighting and CJK phrase
    /// verification data for a further saving of about a tenth.
    pub fn to_binary(&self, include_positions: bool) -> Vec<u8> {
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
    /// This materializes the postings into the same in-memory structures
    /// [`Index::load`] builds, so search behaves identically. It is the
    /// convenient path, not the zero-copy one — use [`crate::BinaryIndex`]
    /// directly to read postings straight out of a mapped buffer.
    ///
    /// The `language` must match the one the index was built with; a mismatch
    /// makes query tokenization disagree with the indexed terms.
    pub fn from_binary(
        bytes: &[u8],
        language: LanguageRef,
    ) -> Result<Self, crate::binary::FormatError> {
        let binary = crate::binary::BinaryIndex::open(bytes)?;

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

        let mut field_lengths: HashMap<FieldRef, usize> = HashMap::new();
        for (doc_id, doc_ref) in doc_refs.iter().enumerate() {
            for (field_id, field_name) in fields.iter().enumerate() {
                let length = binary.field_length(doc_id as u32, field_id as u32)?;
                field_lengths.insert(
                    FieldRef::new(doc_ref.clone(), field_name.clone()),
                    length as usize,
                );
            }
        }

        let mut field_boosts = HashMap::with_capacity(fields.len());
        for (field_id, field_name) in fields.iter().enumerate() {
            field_boosts.insert(field_name.clone(), binary.field_boost(field_id as u32)?);
        }

        let mut doc_boosts = HashMap::with_capacity(doc_refs.len());
        for (doc_id, doc_ref) in doc_refs.iter().enumerate() {
            doc_boosts.insert(doc_ref.clone(), binary.doc_boost(doc_id as u32)?);
        }

        let stats = Stats {
            document_count: binary.document_count(),
            average_field_lengths: average_field_lengths(&field_lengths, &fields),
            field_lengths,
            field_boosts,
            doc_boosts,
            k1: binary.k1(),
            b: binary.b(),
        };

        let token_set = TokenSet::from_sorted(&inverted_index.keys().cloned().collect::<Vec<_>>());

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
}

/// Whether a document satisfies the accumulated set of required clauses.
///
/// `None` means the query had no required clauses, so every document passes.
///
/// Written out rather than using `Option::is_none_or`, which needs Rust 1.82
/// and would raise the crate's 1.78 MSRV.
fn satisfies_required(required: Option<&HashSet<String>>, doc_ref: &str) -> bool {
    match required {
        Some(set) => set.contains(doc_ref),
        None => true,
    }
}

/// Recompute mean field lengths from the per-field-ref lengths.
///
/// The averages are derived on load rather than serialized, since they are a
/// pure function of data already present and would otherwise be one more thing
/// that can disagree with itself.
fn average_field_lengths(
    field_lengths: &HashMap<FieldRef, usize>,
    fields: &[String],
) -> HashMap<String, f64> {
    let mut total: HashMap<&str, usize> = HashMap::new();
    let mut count: HashMap<&str, usize> = HashMap::new();
    for (field_ref, len) in field_lengths {
        *total.entry(field_ref.field_name.as_str()).or_insert(0) += len;
        *count.entry(field_ref.field_name.as_str()).or_insert(0) += 1;
    }
    fields
        .iter()
        .map(|name| {
            let sum = total.get(name.as_str()).copied().unwrap_or(0) as f64;
            let n = count.get(name.as_str()).copied().unwrap_or(1).max(1) as f64;
            (name.clone(), sum / n)
        })
        .collect()
}

/// Serialized index format.
#[derive(Serialize, Deserialize)]
struct SerializedIndex {
    version: String,
    language: String,
    fields: Vec<String>,
    #[serde(rename = "fieldBoosts", default)]
    field_boosts: Vec<f64>,
    #[serde(rename = "documentCount")]
    document_count: usize,
    k1: f64,
    b: f64,
    /// `fieldName/docRef` -> token count.
    #[serde(rename = "fieldLengths")]
    field_lengths: Vec<(String, usize)>,
    #[serde(rename = "docBoosts", default)]
    doc_boosts: Vec<(String, f64)>,
    #[serde(rename = "invertedIndex")]
    inverted_index: Vec<(String, SerializedPosting)>,
    pipeline: Vec<String>,
}

/// Serialized posting for a single term.
#[derive(Serialize, Deserialize)]
struct SerializedPosting {
    #[serde(flatten)]
    fields: HashMap<String, HashMap<String, SerializedPostingDoc>>,
}

/// Serialized per-document posting data.
///
/// Both fields default, so a posting written without positions (a positions-free
/// index) or without an explicit term frequency still deserializes.
#[derive(Serialize, Deserialize)]
struct SerializedPostingDoc {
    #[serde(rename = "tf", default)]
    term_frequency: u32,
    #[serde(rename = "p", default, skip_serializing_if = "Vec::is_empty")]
    positions: Vec<(usize, usize)>,
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
}
