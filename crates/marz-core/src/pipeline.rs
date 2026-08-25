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

    /// Return the serialization labels for this pipeline's functions.
    pub fn labels(&self) -> Vec<String> {
        self.language
            .pipeline_labels()
            .iter()
            .map(|s| s.to_string())
            .collect()
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

    /// Run the search pipeline: tokenize, trim, and stem the input string.
    ///
    /// Returns [`Token`]s rather than bare strings so that callers keep each
    /// term's position in the query. Positions are what make CJK phrase
    /// verification possible: a query like `検索エンジン` becomes several
    /// bigrams, and only their query-side offsets reveal that they were
    /// contiguous in the original input and so should be contiguous in a
    /// document too.
    pub fn run_search(&self, text: &str) -> Vec<Token> {
        let tokens = self.language.tokenize(text);
        let mut output = Vec::new();
        for mut token in tokens {
            if !self.language.trim(&mut token) {
                continue;
            }
            token.term = self.language.stem(&token.term);
            output.push(token);
        }
        output
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
