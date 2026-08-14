//! Language configuration and registry.
//!
//! Marz treats i18n as a core feature. Every locale implements the same
//! [`Language`] trait used by English, so no language is a second-class add-on.

use std::sync::Arc;

use crate::token::Token;

/// Language-specific text processing.
///
/// All languages — including English — implement this trait. The tokenizer,
/// trimmer, stop-word filter, and stemmer are called by the generic pipeline.
pub trait Language: Send + Sync {
    /// ISO-style language code, e.g. "en" or "zh".
    fn code(&self) -> &str;

    /// Tokenize text into tokens.
    fn tokenize(&self, text: &str) -> Vec<Token>;

    /// Trim non-word characters from a token. Return `false` to drop it.
    fn trim(&self, token: &mut Token) -> bool;

    /// Return `true` if the term is a stop word.
    fn is_stop_word(&self, term: &str) -> bool;

    /// Stem a term. Return the term unchanged if no stemming is needed.
    fn stem(&self, term: &str) -> String;
}

/// A language handle used throughout the engine.
pub type LanguageRef = Arc<dyn Language>;

/// Combine several languages into one configuration.
///
/// This follows the lunr-languages approach: each configured tokenizer runs,
/// stop words are unioned, and stemmers are chained.
pub struct MultiLanguage {
    code: String,
    languages: Vec<LanguageRef>,
}

impl MultiLanguage {
    /// Create a multi-language configuration.
    pub fn new(languages: Vec<LanguageRef>) -> Self {
        let code = languages
            .iter()
            .map(|l| l.code())
            .collect::<Vec<_>>()
            .join("-");
        Self { code, languages }
    }
}

impl Language for MultiLanguage {
    fn code(&self) -> &str {
        &self.code
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut seen = std::collections::HashSet::new();
        let mut tokens = Vec::new();
        for lang in &self.languages {
            for token in lang.tokenize(text) {
                if seen.insert(token.term.clone()) {
                    tokens.push(token);
                }
            }
        }
        tokens
    }

    fn trim(&self, token: &mut Token) -> bool {
        // A token passes the trimmer if any configured language keeps it.
        for lang in &self.languages {
            if lang.trim(token) {
                return true;
            }
        }
        false
    }

    fn is_stop_word(&self, term: &str) -> bool {
        self.languages.iter().any(|l| l.is_stop_word(term))
    }

    fn stem(&self, term: &str) -> String {
        let mut result = term.to_string();
        for lang in &self.languages {
            result = lang.stem(&result);
        }
        result
    }
}
