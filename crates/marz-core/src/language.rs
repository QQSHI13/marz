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

    /// Characters that separate query terms in the query lexer.
    ///
    /// These should match the separators used by [`Language::tokenize`].
    fn separator_chars(&self) -> &str {
        " \t\n\r\x0C\x0B\x0D\u{00A0}"
    }

    /// Labels for the indexing pipeline functions, used for serialization.
    ///
    /// An English pipeline returns `["trimmer", "stopWordFilter", "stemmer"]`.
    /// Languages without these functions return an empty list.
    fn pipeline_labels(&self) -> Vec<&'static str> {
        Vec::new()
    }

    /// Return `true` if `c` belongs to a script this language tokenizes into
    /// overlapping n-grams.
    ///
    /// Query handling needs this. A query term in an n-grammed script expands
    /// into several bigrams that must be *adjacent* in a document to count as a
    /// phrase match, whereas a word-tokenized term stands alone. Without this
    /// hook the query layer cannot tell the two cases apart.
    fn is_ngram_script(&self, _c: char) -> bool {
        false
    }
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
        // Deduplicate on (term, position) rather than term alone. Deduplicating
        // on the term would collapse repeated words in a document down to one
        // occurrence, destroying both the term frequency and the positions that
        // CJK phrase matching depends on.
        let mut seen = std::collections::HashSet::new();
        let mut tokens = Vec::new();
        for lang in &self.languages {
            for token in lang.tokenize(text) {
                let key = (token.term.clone(), token.position());
                if seen.insert(key) {
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

    fn is_ngram_script(&self, c: char) -> bool {
        self.languages.iter().any(|l| l.is_ngram_script(c))
    }
}
