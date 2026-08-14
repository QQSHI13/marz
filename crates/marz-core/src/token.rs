//! Tokens carry a string and metadata through the pipeline.

use std::collections::HashMap;

/// Metadata key for token position: `(start, length)` in the source text.
pub const POSITION: &str = "position";

/// Metadata key for token index within its document field.
pub const INDEX: &str = "index";

/// A token wraps a string and arbitrary metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// The token string.
    pub term: String,
    /// Associated metadata, e.g. position and index.
    pub metadata: HashMap<String, TokenMetadata>,
}

/// Metadata values that can be attached to a token.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TokenMetadata {
    /// A single integer value.
    Integer(usize),
    /// A pair of integer values, e.g. `(start, length)`.
    Pair(usize, usize),
    /// A string value.
    String(String),
}

impl Token {
    /// Create a new token with the given term.
    pub fn new(term: impl Into<String>) -> Self {
        Self {
            term: term.into(),
            metadata: HashMap::new(),
        }
    }

    /// Create a token with position and index metadata, matching lunr's defaults.
    pub fn with_position(
        term: impl Into<String>,
        start: usize,
        length: usize,
        index: usize,
    ) -> Self {
        let mut token = Self::new(term);
        token
            .metadata
            .insert(POSITION.to_string(), TokenMetadata::Pair(start, length));
        token
            .metadata
            .insert(INDEX.to_string(), TokenMetadata::Integer(index));
        token
    }

    /// Update the token term in place.
    pub fn update(&mut self, f: impl FnOnce(&str) -> String) -> &mut Self {
        self.term = f(&self.term);
        self
    }
}

impl From<&str> for Token {
    fn from(term: &str) -> Self {
        Token::new(term)
    }
}

impl From<String> for Token {
    fn from(term: String) -> Self {
        Token::new(term)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_position_metadata() {
        let t = Token::with_position("hello", 0, 5, 0);
        assert_eq!(t.term, "hello");
        assert_eq!(t.metadata.get(POSITION), Some(&TokenMetadata::Pair(0, 5)));
        assert_eq!(t.metadata.get(INDEX), Some(&TokenMetadata::Integer(0)));
    }
}
