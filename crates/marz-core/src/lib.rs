//! Marz core search engine.
//!
//! A Rust implementation of a lunr-compatible offline search index with
//! first-class internationalization support.

#![warn(missing_docs)]

pub mod index;
pub mod language;
pub mod languages;
pub mod normalize;
pub mod pipeline;
pub mod query;
pub mod query_parser;
pub mod token;
pub mod token_set;
pub mod tokenizer;

/// Re-export core types.
pub use index::{Index, IndexBuilder, MatchData, SearchResult};
pub use language::Language;
pub use query::Query;
pub use token::Token;

/// Compute the lunr IDF for a term given its document frequency and total docs.
///
/// Formula: log(1 + abs((N - df + 0.5) / (df + 0.5)))
pub fn idf(document_count: usize, doc_frequency: usize) -> f64 {
    let n = document_count as f64;
    let df = doc_frequency as f64;
    let x = (n - df + 0.5) / (df + 0.5);
    (1.0 + x.abs()).ln()
}

/// Compute the BM25 weight for one term occurrence in one document field.
///
/// This is evaluated at *query* time, once per matching posting, rather than
/// precomputed into a stored field vector. See [`index`] for why.
///
/// Formula:
/// w = idf * ((k1 + 1) * tf) / (k1 * (1 - b + b * (field_len / avg_field_len)) + tf)
/// w *= field_boost * doc_boost
/// w = round(w, 3)
#[allow(clippy::too_many_arguments)]
pub fn bm25_weight(
    idf: f64,
    tf: f64,
    field_len: f64,
    avg_field_len: f64,
    k1: f64,
    b: f64,
    field_boost: f64,
    doc_boost: f64,
) -> f64 {
    let denom = k1 * (1.0 - b + b * (field_len / avg_field_len)) + tf;
    let score = idf * ((k1 + 1.0) * tf) / denom;
    let score = score * field_boost * doc_boost;
    (score * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idf() {
        // When df is half of N, x = 1, idf = log(2)
        let result = idf(100, 50);
        assert!((result - 2.0f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn test_bm25_weight_rounding() {
        let w = bm25_weight(1.5, 2.0, 10.0, 10.0, 1.2, 0.75, 1.0, 1.0);
        // Rounded to 3 decimal places
        assert_eq!((w * 1000.0).round() / 1000.0, w);
    }
}
