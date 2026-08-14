//! Pipeline applies language-specific transformations to tokens.

use crate::language::LanguageRef;
use crate::token::Token;

/// A pipeline holds a list of functions applied to tokens.
pub struct Pipeline {
    /// Language configuration used by the pipeline.
    language: LanguageRef,
}

impl Pipeline {
    /// Create a pipeline for the given language.
    pub fn new(language: LanguageRef) -> Self {
        Self { language }
    }

    /// Return the language used by this pipeline.
    pub fn language(&self) -> LanguageRef {
        self.language.clone()
    }

    /// Run the default indexing pipeline: trim, stop-word filter, stem.
    pub fn run_index(&self, tokens: Vec<Token>) -> Vec<Token> {
        let mut output = Vec::new();
        for mut token in tokens {
            // Trimmer
            if !self.language.trim(&mut token) {
                continue;
            }
            // Stop-word filter
            if self.language.is_stop_word(&token.term) {
                continue;
            }
            // Stemmer
            let stemmed = self.language.stem(&token.term);
            token.term = stemmed;
            output.push(token);
        }
        output
    }

    /// Run the search pipeline: stem only.
    pub fn run_search(&self, mut token: Token) -> Vec<String> {
        if !self.language.trim(&mut token) {
            return Vec::new();
        }
        vec![self.language.stem(&token.term)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::languages::English;
    use std::sync::Arc;

    #[test]
    fn english_index_pipeline() {
        let lang: LanguageRef = Arc::new(English);
        let pipeline = Pipeline::new(lang);
        let tokens = vec![
            Token::with_position("the", 0, 3, 0),
            Token::with_position("running", 4, 7, 1),
        ];
        let result = pipeline.run_index(tokens);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].term, "run");
    }
}
