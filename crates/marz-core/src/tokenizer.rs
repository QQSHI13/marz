//! Default separator-based tokenizer used by most languages.

use crate::token::Token;

/// Tokenize text by splitting on separator characters.
///
/// `separators` is a string where each character is considered a separator.
/// Tokens are lowercased and annotated with position metadata.
pub fn tokenize_with_separator(text: &str, separators: &str) -> Vec<Token> {
    let text_lower = text.to_lowercase();
    let chars: Vec<char> = text_lower.chars().collect();
    let mut tokens = Vec::new();
    let mut slice_start = 0;

    for (i, &ch) in chars.iter().enumerate() {
        if separators.contains(ch) {
            let slice_len = i - slice_start;
            if slice_len > 0 {
                let term: String = chars[slice_start..i].iter().collect();
                tokens.push(Token::with_position(
                    term,
                    slice_start,
                    slice_len,
                    tokens.len(),
                ));
            }
            slice_start = i + 1;
        }
    }

    // End of string
    let slice_len = chars.len() - slice_start;
    if slice_len > 0 {
        let term: String = chars[slice_start..].iter().collect();
        tokens.push(Token::with_position(
            term,
            slice_start,
            slice_len,
            tokens.len(),
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
}
