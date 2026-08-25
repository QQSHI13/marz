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
    ///
    /// Position metadata is left alone, so this is for transformations that do
    /// not change which part of the source the token covers — stemming, folding.
    /// For trimming, use [`Token::trim_matching`], which keeps the position in
    /// step with the term.
    pub fn update(&mut self, f: impl FnOnce(&str) -> String) -> &mut Self {
        self.term = f(&self.term);
        self
    }

    /// Strip leading and trailing characters that fail `keep`, adjusting the
    /// recorded position to match.
    ///
    /// Returns `false` if nothing is left, meaning the caller should drop the
    /// token.
    ///
    /// Two things this gets right that the obvious implementations do not.
    ///
    /// It adjusts the position. Trimming through [`Token::update`] leaves the
    /// position covering the untrimmed span, so `keyboard.` trims to the term
    /// `keyboard` while still reporting a length of nine. Nothing in scoring
    /// notices — positions are read only for highlighting and CJK phrase
    /// verification — which is why it stayed hidden: every highlight silently
    /// included the punctuation the trimmer had just removed.
    ///
    /// And it works in characters. A byte-indexed version (`s[start..end]` from
    /// `rfind(..).map(|i| i + 1)`) panics on any token whose last kept character
    /// is multi-byte, which real CJK text produces constantly via stray Greek
    /// letters, accented Latin and the Japanese iteration mark `々`.
    pub fn trim_matching(&mut self, keep: impl Fn(char) -> bool) -> bool {
        let chars: Vec<char> = self.term.chars().collect();
        let start = chars.iter().position(|c| keep(*c)).unwrap_or(chars.len());
        let end = chars
            .iter()
            .rposition(|c| keep(*c))
            .map(|i| i + 1)
            .unwrap_or(start);

        self.term = chars[start..end].iter().collect();
        if self.term.is_empty() {
            return false;
        }

        if let Some(TokenMetadata::Pair(position_start, _)) = self.metadata.get(POSITION) {
            let trimmed = TokenMetadata::Pair(position_start + start, end - start);
            self.metadata.insert(POSITION.to_string(), trimmed);
        }
        true
    }

    /// Return the `(start, length)` position metadata, if present.
    ///
    /// Positions are measured in characters, not bytes, so they stay valid for
    /// CJK text.
    pub fn position(&self) -> Option<(usize, usize)> {
        match self.metadata.get(POSITION) {
            Some(TokenMetadata::Pair(start, len)) => Some((*start, *len)),
            _ => None,
        }
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

    fn is_word_char(c: char) -> bool {
        c.is_alphanumeric() || c == '_'
    }

    #[test]
    fn trimming_moves_the_position_with_the_term() {
        // "no keyboard." — the token starts at 3 and the period is trimmed, so
        // the position must shrink to cover "keyboard" and nothing else.
        let mut token = Token::with_position("keyboard.", 3, 9, 1);
        assert!(token.trim_matching(is_word_char));
        assert_eq!(token.term, "keyboard");
        assert_eq!(token.position(), Some((3, 8)));
    }

    #[test]
    fn trimming_the_front_advances_the_start() {
        let mut token = Token::with_position("(rust)", 10, 6, 0);
        assert!(token.trim_matching(is_word_char));
        assert_eq!(token.term, "rust");
        assert_eq!(token.position(), Some((11, 4)));
    }

    #[test]
    fn a_token_with_nothing_to_keep_is_dropped() {
        let mut token = Token::with_position("---", 0, 3, 0);
        assert!(!token.trim_matching(is_word_char));
    }

    #[test]
    fn trimming_counts_characters_not_bytes() {
        // Every one of these panicked in a byte-indexed version, because the
        // last kept character is multi-byte.
        for input in ["«café»", "”é”", "--é--", "(π)", "々"] {
            let mut token = Token::with_position(input, 0, input.chars().count(), 0);
            let kept = token.trim_matching(is_word_char);
            if kept {
                let (start, length) = token.position().unwrap();
                // The reported span must still describe the term it kept.
                assert_eq!(length, token.term.chars().count());
                assert!(start + length <= input.chars().count());
            }
        }
    }

    #[test]
    fn a_token_needing_no_trim_keeps_its_position() {
        let mut token = Token::with_position("rust", 7, 4, 0);
        assert!(token.trim_matching(is_word_char));
        assert_eq!(token.position(), Some((7, 4)));
    }

    #[test]
    fn trimming_a_token_without_a_position_does_not_invent_one() {
        let mut token = Token::new("rust.");
        assert!(token.trim_matching(is_word_char));
        assert_eq!(token.term, "rust");
        assert_eq!(token.position(), None);
    }
}
