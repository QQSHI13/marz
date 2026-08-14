//! Index builder and searcher.
//!
//! The builder replicates lunr's `lunr.Builder`:
//!
//! * tokenizes and pipelines each document field,
//! * builds an inverted index that maps terms to `(field, document)` postings,
//! * computes per-field BM25 vectors,
//! * builds a trie of all index terms for wildcard expansion.
//!
//! The searcher scores documents using lunr's asymmetric cosine similarity
//! between a per-field query vector and the stored field vectors.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::language::LanguageRef;
use crate::pipeline::Pipeline;
use crate::query::{Presence, Query};
use crate::query_parser::{parse_query, QueryParseError};
use crate::token_set::TokenSet;
use crate::vector::Vector;
use crate::{bm25_weight, idf};

/// A reference to one field of one document, formatted as `docRef/fieldName`.
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
}

impl std::fmt::Display for FieldRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.doc_ref, self.field_name)
    }
}

/// Metadata stored for a single term occurrence in a document field.
///
/// The default build keeps this empty to stay compact; position metadata can be
/// added later when needed for snippets.
#[derive(Debug, Clone, Default)]
pub struct PostingDoc {
    /// Term positions in the field, as `(start, length)` byte offsets.
    pub positions: Vec<(usize, usize)>,
}

/// Inverted-index posting for a single term.
#[derive(Debug, Clone, Default)]
pub struct Posting {
    /// Dense vector-space index assigned to this term.
    pub index: usize,
    /// Maps field name -> document reference -> posting metadata.
    pub fields: HashMap<String, HashMap<String, PostingDoc>>,
}

/// A matching term/field combination returned with a search result.
#[derive(Debug, Clone, Default)]
pub struct MatchData {
    /// Maps a matched term to the set of fields it matched in.
    pub terms: HashMap<String, HashSet<String>>,
}

impl MatchData {
    /// Record that `term` matched in `field`.
    pub fn add_term(&mut self, term: &str, field: &str) {
        self.terms
            .entry(term.to_string())
            .or_default()
            .insert(field.to_string());
    }

    /// Merge another match data object into this one.
    pub fn merge(&mut self, other: &Self) {
        for (term, fields) in &other.terms {
            let entry = self.terms.entry(term.clone()).or_default();
            for field in fields {
                entry.insert(field.clone());
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

/// Index builder.
pub struct IndexBuilder {
    language: LanguageRef,
    ref_field: String,
    fields: Vec<FieldConfig>,
    field_map: HashMap<String, usize>,
    field_term_frequencies: HashMap<FieldRef, HashMap<String, usize>>,
    field_lengths: HashMap<FieldRef, usize>,
    inverted_index: BTreeMap<String, Posting>,
    document_count: usize,
    doc_boosts: HashMap<String, f64>,
    k1: f64,
    b: f64,
    term_index_counter: usize,
}

impl IndexBuilder {
    /// Create a new builder for the given language.
    pub fn new(language: LanguageRef) -> Self {
        Self {
            language,
            ref_field: "id".to_string(),
            fields: Vec::new(),
            field_map: HashMap::new(),
            field_term_frequencies: HashMap::new(),
            field_lengths: HashMap::new(),
            inverted_index: BTreeMap::new(),
            document_count: 0,
            doc_boosts: HashMap::new(),
            k1: 1.2,
            b: 0.75,
            term_index_counter: 0,
        }
    }

    /// Set the document reference field. Defaults to `"id"`.
    pub fn ref_field(&mut self, field: impl Into<String>) -> &mut Self {
        self.ref_field = field.into();
        self
    }

    /// Add a field to the index. `boost` defaults to `1.0`.
    pub fn field(&mut self, name: impl Into<String>, boost: f64) -> &mut Self {
        let name = name.into();
        self.field_map.insert(name.clone(), self.fields.len());
        self.fields.push(FieldConfig {
            name,
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
            self.field_lengths.insert(field_ref.clone(), terms.len());

            let mut term_frequencies: HashMap<String, usize> = HashMap::new();

            for term in terms {
                *term_frequencies.entry(term.term.clone()).or_insert(0) += 1;

                let posting = self
                    .inverted_index
                    .entry(term.term.clone())
                    .or_insert_with(|| {
                        let mut posting = Posting {
                            index: self.term_index_counter,
                            fields: HashMap::new(),
                        };
                        self.term_index_counter += 1;
                        for f in &self.fields {
                            posting.fields.insert(f.name.clone(), HashMap::new());
                        }
                        posting
                    });

                posting
                    .fields
                    .get_mut(&field.name)
                    .expect("field initialized when posting created")
                    .entry(doc_ref.clone())
                    .or_default();
            }

            if !term_frequencies.is_empty() {
                self.field_term_frequencies
                    .insert(field_ref, term_frequencies);
            }
        }
    }

    /// Consume the builder and produce a searchable [`Index`].
    pub fn build(self) -> Index {
        let average_field_lengths = self.calculate_average_field_lengths();
        let field_vectors = self.create_field_vectors(&average_field_lengths);
        let token_set =
            TokenSet::from_sorted(&self.inverted_index.keys().cloned().collect::<Vec<_>>());

        Index {
            inverted_index: self.inverted_index,
            field_vectors,
            token_set,
            fields: self.fields.iter().map(|f| f.name.clone()).collect(),
            pipeline: Pipeline::new(self.language),
        }
    }

    fn calculate_average_field_lengths(&self) -> HashMap<String, f64> {
        let mut accumulator: HashMap<String, usize> = HashMap::new();
        let mut documents_with_field: HashMap<String, usize> = HashMap::new();

        for (field_ref, length) in &self.field_lengths {
            *accumulator.entry(field_ref.field_name.clone()).or_insert(0) += length;
            *documents_with_field
                .entry(field_ref.field_name.clone())
                .or_insert(0) += 1;
        }

        let mut average: HashMap<String, f64> = HashMap::new();
        for field in &self.fields {
            let total = accumulator.get(&field.name).copied().unwrap_or(0) as f64;
            let count = documents_with_field
                .get(&field.name)
                .copied()
                .unwrap_or(1)
                .max(1) as f64;
            average.insert(field.name.clone(), total / count);
        }
        average
    }

    fn create_field_vectors(
        &self,
        average_field_lengths: &HashMap<String, f64>,
    ) -> HashMap<FieldRef, Vector> {
        let mut field_vectors = HashMap::new();

        for (field_ref, term_frequencies) in &self.field_term_frequencies {
            let field_name = &field_ref.field_name;
            let field_length = *self.field_lengths.get(field_ref).unwrap_or(&0) as f64;
            let average_length = *average_field_lengths.get(field_name).unwrap_or(&0.0);
            let field_boost = self
                .fields
                .get(*self.field_map.get(field_name).unwrap_or(&0))
                .map(|f| f.boost)
                .unwrap_or(1.0);
            let doc_boost = *self.doc_boosts.get(&field_ref.doc_ref).unwrap_or(&1.0);

            let mut vector = Vector::new();
            for (term, tf) in term_frequencies {
                let posting = self
                    .inverted_index
                    .get(term)
                    .expect("term in frequencies must be indexed");
                let term_index = posting.index;
                let document_frequency: usize =
                    posting.fields.values().map(|docs| docs.len()).sum();
                let idf_value = idf(self.document_count, document_frequency);
                let weight = bm25_weight(
                    idf_value,
                    *tf as f64,
                    field_length,
                    average_length,
                    self.k1,
                    self.b,
                    field_boost,
                    doc_boost,
                );
                vector.insert(term_index, weight);
            }

            field_vectors.insert(field_ref.clone(), vector);
        }

        field_vectors
    }
}

/// Built search index.
pub struct Index {
    inverted_index: BTreeMap<String, Posting>,
    field_vectors: HashMap<FieldRef, Vector>,
    token_set: TokenSet,
    fields: Vec<String>,
    pipeline: Pipeline,
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

    fn execute_query(&self, query: &Query) -> Vec<SearchResult> {
        let mut query_vectors: HashMap<String, Vector> = HashMap::new();
        for field in &self.fields {
            query_vectors.insert(field.clone(), Vector::new());
        }

        let mut matching_fields: HashMap<FieldRef, MatchData> = HashMap::new();
        let mut required_matches: HashMap<String, Option<HashSet<String>>> = HashMap::new();
        let mut prohibited_matches: HashMap<String, HashSet<String>> = HashMap::new();

        for clause in &query.clauses {
            let terms = if clause.use_pipeline {
                self.pipeline.run_search(&clause.term)
            } else {
                vec![clause.term.clone()]
            };

            let mut clause_matches: HashSet<String> = HashSet::new();
            let mut short_circuit = false;

            for term in terms {
                if term.is_empty() {
                    continue;
                }

                let expanded = if clause.term.contains('*') {
                    self.token_set.expand(&term)
                } else if let Some(edits) = clause.edit_distance {
                    self.token_set.expand_fuzzy(&term, edits)
                } else {
                    self.token_set.expand(&term)
                };

                if expanded.is_empty() {
                    if clause.presence == Presence::Required {
                        for field in &clause.fields {
                            required_matches.insert(field.clone(), Some(HashSet::new()));
                        }
                        short_circuit = true;
                    }
                    continue;
                }

                if short_circuit {
                    continue;
                }

                for expanded_term in expanded {
                    let Some(posting) = self.inverted_index.get(&expanded_term) else {
                        continue;
                    };
                    let term_index = posting.index;

                    for field_name in &clause.fields {
                        query_vectors.get_mut(field_name).unwrap().upsert(
                            term_index,
                            clause.boost,
                            |a, b| a + b,
                        );

                        let docs: HashSet<String> = posting
                            .fields
                            .get(field_name)
                            .map(|m| m.keys().cloned().collect())
                            .unwrap_or_default();

                        match clause.presence {
                            Presence::Required => {
                                clause_matches.extend(docs.iter().cloned());
                            }
                            Presence::Prohibited => {
                                prohibited_matches
                                    .entry(field_name.clone())
                                    .or_default()
                                    .extend(docs.iter().cloned());
                                continue;
                            }
                            Presence::Optional => {}
                        }

                        if let Some(posting_field) = posting.fields.get(field_name) {
                            for doc_ref in posting_field.keys() {
                                let field_ref = FieldRef::new(doc_ref, field_name);
                                matching_fields
                                    .entry(field_ref)
                                    .or_default()
                                    .add_term(&expanded_term, field_name);
                            }
                        }
                    }
                }
            }

            if clause.presence == Presence::Required && !short_circuit {
                for field_name in &clause.fields {
                    required_matches
                        .entry(field_name.clone())
                        .and_modify(|opt| {
                            if let Some(set) = opt {
                                set.retain(|d| clause_matches.contains(d));
                            }
                        })
                        .or_insert_with(|| Some(clause_matches.clone()));
                }
            }
        }

        // Combine field-scoped required and prohibited sets into global sets.
        let mut all_required: Option<HashSet<String>> = None;
        for field in &self.fields {
            if let Some(Some(field_required)) = required_matches.get(field) {
                all_required = Some(match all_required {
                    Some(acc) => acc.intersection(field_required).cloned().collect(),
                    None => field_required.clone(),
                });
            }
        }

        let mut all_prohibited: HashSet<String> = HashSet::new();
        for field in &self.fields {
            if let Some(field_prohibited) = prohibited_matches.get(field) {
                all_prohibited.extend(field_prohibited.iter().cloned());
            }
        }

        let is_negated = query.is_negated();
        let matching_field_refs: Vec<FieldRef> = if is_negated {
            self.field_vectors.keys().cloned().collect()
        } else {
            matching_fields.keys().cloned().collect()
        };

        let mut results: HashMap<String, SearchResult> = HashMap::new();
        for field_ref in matching_field_refs {
            let doc_ref = field_ref.doc_ref.clone();

            if !all_required.as_ref().map_or(true, |s| s.contains(&doc_ref)) {
                continue;
            }
            if all_prohibited.contains(&doc_ref) {
                continue;
            }

            let query_vector = query_vectors.get(&field_ref.field_name).unwrap();
            let field_vector = self
                .field_vectors
                .get(&field_ref)
                .expect("matching field must have a vector");
            let score = query_vector.similarity(field_vector);
            if score == 0.0 && !is_negated {
                continue;
            }

            let match_data = if is_negated {
                MatchData::default()
            } else {
                matching_fields.get(&field_ref).cloned().unwrap_or_default()
            };

            match results.get_mut(&doc_ref) {
                Some(result) => {
                    result.score += score;
                    result.match_data.merge(&match_data);
                }
                None => {
                    results.insert(
                        doc_ref.clone(),
                        SearchResult {
                            ref_id: doc_ref,
                            score,
                            match_data,
                        },
                    );
                }
            }
        }

        let mut results: Vec<SearchResult> = results.into_values().collect();
        results.sort_by(|a, b| {
            let by_score = b
                .score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal);
            by_score.then_with(|| a.ref_id.cmp(&b.ref_id))
        });
        results
    }
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
    fn index_builds_vectors() {
        let index = test_index();
        assert!(!index.field_vectors.is_empty());
        assert!(!index.inverted_index.is_empty());
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
        assert!(results.iter().any(|r| r.ref_id == "a"));
        assert!(results.iter().any(|r| r.ref_id == "b"));
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
}
