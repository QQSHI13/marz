//! Default separator-based tokenizer used by most languages.

use crate::normalize::normalize;
use crate::token::Token;

/// Whether `c` is a word character for trimming: alphanumeric or `_`.
#[inline]
pub fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Tokenize text by splitting on separator characters.
///
/// `separators` is a string where each character is considered a separator.
/// Text is normalized (see [`crate::normalize`]) — which folds full-width
/// Latin and lowercases — and tokens are annotated with `(start, length)`
/// positions measured in **characters**, not bytes, so they remain valid for
/// non-ASCII text.
///
/// Positions are offsets into the **normalized** text, not the original input:
/// normalization is not length-preserving (`ｶﾞ` is two code points becoming
/// one `ガ`), so highlighting must normalize the field text first (see the
/// `normalize` export in the Python/WASM bindings).
///
/// `normalize` is idempotent, so callers that have already normalized their
/// input (such as the CJK tokenizer, which normalizes the whole string before
/// splitting it into script runs) may call [`tokenize_normalized`] to avoid a
/// second pass.
pub fn tokenize_with_separator(text: &str, separators: &str) -> Vec<Token> {
    tokenize_normalized(&normalize(text), separators)
}

/// Tokenize already-normalized text, skipping the second normalize pass.
///
/// `text` must already be [`normalize`]d. Used by the CJK tokenizer for Latin
/// runs sliced out of an already-normalized string.
pub fn tokenize_normalized(normalized: &str, separators: &str) -> Vec<Token> {
    let chars: Vec<char> = normalized.chars().collect();
    // Fast path for the common ASCII separators: a 128-entry table instead of
    // `str::contains` (O(S)) per character.
    let mut ascii_sep = [false; 128];
    let mut has_non_ascii_sep = false;
    for c in separators.chars() {
        if (c as u32) < 128 {
            ascii_sep[c as usize] = true;
        } else {
            has_non_ascii_sep = true;
        }
    }
    let is_sep = |ch: char| {
        if (ch as u32) < 128 {
            ascii_sep[ch as usize]
        } else if has_non_ascii_sep {
            separators.contains(ch)
        } else {
            false
        }
    };

    let mut tokens = Vec::new();
    let mut slice_start = 0;

    for (i, &ch) in chars.iter().enumerate() {
        if is_sep(ch) {
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
