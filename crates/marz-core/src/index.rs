//! Index builder and searcher.

use crate::language::LanguageRef;

/// Index builder.
pub struct IndexBuilder {
    language: LanguageRef,
}

impl IndexBuilder {
    /// Create a new builder for the given language.
    pub fn new(language: LanguageRef) -> Self {
        Self { language }
    }
}

/// Built search index.
pub struct Index;
