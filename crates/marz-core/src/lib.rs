//! Marz core search engine.
//!
//! A dependency-light offline search index with first-class CJK support,
//! achieved with n-gram tokenization rather than a segmentation dictionary.
//!
//! An index can be serialized as JSON (with the `json` cargo feature, on by
//! default), or as the compact zero-copy [`binary`] format, which is roughly
//! a fifth of the size and can be read straight from a memory-mapped file.

#![warn(missing_docs)]

pub mod binary;
pub mod index;
pub mod language;
pub mod languages;
pub mod normalize;
pub mod phrase;
pub mod pipeline;
pub mod query;
pub mod query_parser;
pub mod stemmers;
pub mod token;
pub mod token_set;
pub mod tokenizer;

/// Re-export core types.
pub use binary::BinaryIndex;
pub use index::{Index, IndexBuilder, MatchData, SearchResult};
pub use language::{Language, MultiLanguage};
pub use query::Query;
pub use token::Token;

/// Compute the lunr IDF for a term given its document frequency and total docs.
///
/// Formula: log(1 + abs((N - df + 0.5) / (df + 0.5)))
pub fn idf(document_count: usize, doc_frequency: usize) -> f64 {
    debug_assert!(
        doc_frequency <= document_count,
        "df ({doc_frequency}) > N ({document_count}): corrupt index"
    );
    let n = document_count as f64;
    // Clamp: a corrupt index with df > N would otherwise be masked by `abs`
    // into a plausible-looking but wrong ranking.
    let df = (doc_frequency.min(document_count)) as f64;
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
    // Public function: callers other than `Index::score_term` offer no guard,
    // and without this `avg == 0` yields `inf`/`NaN` scores.
    if !tf.is_finite() || !avg_field_len.is_finite() || tf <= 0.0 || avg_field_len <= 0.0 {
        return 0.0;
    }
    if !idf.is_finite() || !field_boost.is_finite() || !doc_boost.is_finite() {
        return 0.0;
    }
    let k1 = if k1.is_finite() && k1 >= 0.0 { k1 } else { 1.2 };
    let b = if b.is_finite() {
        b.clamp(0.0, 1.0)
    } else {
        0.75
    };
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
