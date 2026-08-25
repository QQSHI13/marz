//! English language implementation.

use crate::language::Language;
use crate::languages::porter;
use crate::token::Token;
use crate::tokenizer::tokenize_with_separator;

/// English language configuration.
#[derive(Debug, Clone, Default)]
pub struct English;

impl Language for English {
    fn code(&self) -> &str {
        "en"
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        tokenize_with_separator(text, " \t\n\r\x0C\x0B\x0D\u{00A0}-")
    }

    fn trim(&self, token: &mut Token) -> bool {
        token.update(|s| {
            let chars: Vec<char> = s.chars().collect();
            let start = chars
                .iter()
                .position(|c| is_word_char(*c))
                .unwrap_or(chars.len());
            let end = chars
                .iter()
                .rposition(|c| is_word_char(*c))
                .map(|i| i + 1)
                .unwrap_or(start);
            chars[start..end].iter().collect()
        });
        !token.term.is_empty()
    }

    fn is_stop_word(&self, term: &str) -> bool {
        STOP_WORDS.contains(&term)
    }

    fn stem(&self, term: &str) -> String {
        porter::stem(term)
    }

    fn separator_chars(&self) -> &str {
        " \t\n\r\x0C\x0B\x0D\u{00A0}-"
    }

    fn pipeline_labels(&self) -> Vec<&'static str> {
        vec!["trimmer", "stopWordFilter", "stemmer"]
    }
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// English stop-word list from lunr.js.
const STOP_WORDS: &[&str] = &[
    "a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could",
    "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got",
    "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in",
    "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might",
    "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or",
    "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so", "some",
    "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to",
    "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "yet", "you", "your",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_stemming() {
        let en = English;
        assert_eq!(en.stem("running"), "run");
        assert_eq!(en.stem("flies"), "fli");
        assert_eq!(en.stem("died"), "di");
        assert_eq!(en.stem("national"), "nation");
    }

    #[test]
    fn english_stop_words() {
        let en = English;
        assert!(en.is_stop_word("the"));
        assert!(!en.is_stop_word("marz"));
    }
}
