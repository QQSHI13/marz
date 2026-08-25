//! Default separator-based tokenizer used by most languages.

use crate::normalize::normalize;
use crate::token::Token;

/// Tokenize text by splitting on separator characters.
///
/// `separators` is a string where each character is considered a separator.
/// Text is normalized (see [`crate::normalize`]) — which folds full-width
/// Latin and lowercases — and tokens are annotated with `(start, length)`
/// positions measured in **characters**, not bytes, so they remain valid for
/// non-ASCII text.
///
/// `normalize` is idempotent, so callers that have already normalized their
/// input (such as the CJK tokenizer, which normalizes the whole string before
/// splitting it into script runs) may call this safely.
pub fn tokenize_with_separator(text: &str, separators: &str) -> Vec<Token> {
    let normalized = normalize(text);
    let chars: Vec<char> = normalized.chars().collect();
    let mut tokens = Vec::new();
    let mut slice_start = 0;

    for (i, &ch) in chars.iter().enumerate() {
        if separators.contains(ch) {
            if i > slice_start {
                let term: String = chars[slice_start..i].iter().collect();
                let index = tokens.len();
                tokens.push(Token::with_position(
                    term,
                    slice_start,
                    i - slice_start,
                    index,
                ));
            }
            slice_start = i + 1;
        }
    }

    if chars.len() > slice_start {
        let term: String = chars[slice_start..].iter().collect();
        let index = tokens.len();
        tokens.push(Token::with_position(
            term,
            slice_start,
            chars.len() - slice_start,
            index,
        ));
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_tokenization() {
        let tokens = tokenize_with_separator("Hello world!", " \t\n\r-");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].term, "hello");
        assert_eq!(tokens[1].term, "world!");
    }

    #[test]
    fn positions_are_char_offsets_not_byte_offsets() {
        let tokens = tokenize_with_separator("café latte", " ");
        assert_eq!(tokens[0].term, "café");
        // "café" is 5 bytes but 4 characters, so "latte" starts at char 5.
        assert_eq!(tokens[1].position(), Some((5, 5)));
    }

    #[test]
    fn normalization_applies() {
        let tokens = tokenize_with_separator("ＲＵＳＴ", " ");
        assert_eq!(tokens[0].term, "rust");
    }
}
