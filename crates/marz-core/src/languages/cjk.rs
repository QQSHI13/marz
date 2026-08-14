//! Shared utilities for CJK language tokenizers.

use crate::token::{Token, TokenMetadata, POSITION};
use crate::tokenizer::tokenize_with_separator;

/// Returns true for CJK Unified Ideographs (Han).
pub fn is_cjk_ideograph(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0x2CEB0..=0x2EBEF
            | 0x30000..=0x3134F
    )
}

/// Returns true for Hiragana.
pub fn is_hiragana(c: char) -> bool {
    matches!(c as u32, 0x3040..=0x309F)
}

/// Returns true for Katakana (including half-width).
pub fn is_katakana(c: char) -> bool {
    matches!(c as u32, 0x30A0..=0x30FF | 0xFF65..=0xFF9F)
}

/// Returns true for Hangul syllables and Jamo.
pub fn is_hangul(c: char) -> bool {
    matches!(
        c as u32,
        0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F | 0xA960..=0xA97F | 0xD7B0..=0xD7FF
    )
}

/// Returns true for any CJK script character handled by this module.
pub fn is_cjk_char(c: char) -> bool {
    is_cjk_ideograph(c) || is_hiragana(c) || is_katakana(c) || is_hangul(c)
}

/// Tokenize text using script-aware CJK bigrams plus non-CJK separator tokens.
///
/// For runs of characters matching `is_script`, the tokenizer emits both
/// overlapping bigrams and single-character unigrams. This gives reasonable
/// recall for unknown words (unigrams) while preserving adjacency information
/// for common compounds (bigrams).
///
/// Non-script runs are tokenized with the standard separator tokenizer.
pub fn tokenize_cjk(text: &str, is_script: impl Fn(char) -> bool) -> Vec<Token> {
    let lower = text.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        if is_script(chars[i]) {
            let run_start = i;
            while i < chars.len() && is_script(chars[i]) {
                i += 1;
            }
            let run: String = chars[run_start..i].iter().collect();
            let run_chars: Vec<char> = run.chars().collect();

            // Unigrams.
            for (offset, ch) in run_chars.iter().enumerate() {
                tokens.push(Token::with_position(
                    ch.to_string(),
                    run_start + offset,
                    1,
                    tokens.len(),
                ));
            }

            // Overlapping bigrams.
            for offset in 0..run_chars.len().saturating_sub(1) {
                let bigram: String = run_chars[offset..offset + 2].iter().collect();
                tokens.push(Token::with_position(
                    bigram,
                    run_start + offset,
                    2,
                    tokens.len(),
                ));
            }
        } else {
            let run_start = i;
            while i < chars.len() && !is_script(chars[i]) {
                i += 1;
            }
            let run: String = chars[run_start..i].iter().collect();
            let local_tokens = tokenize_with_separator(&run, " \t\n\r\x0C\x0B\x0D\u{00A0}-");
            for mut t in local_tokens {
                let pos = t
                    .metadata
                    .remove(POSITION)
                    .and_then(|m| match m {
                        TokenMetadata::Pair(start, len) => Some((start, len)),
                        _ => None,
                    })
                    .unwrap_or((0, t.term.len()));
                tokens.push(Token::with_position(
                    t.term,
                    run_start + pos.0,
                    pos.1,
                    tokens.len(),
                ));
            }
        }
    }

    tokens
}

/// Trimmer suitable for CJK languages: CJK terms are kept as-is, ASCII terms
/// are stripped of surrounding non-word characters.
pub fn cjk_trim(token: &mut Token) -> bool {
    if token.term.chars().next().is_some_and(is_cjk_char) {
        !token.term.is_empty()
    } else {
        token.update(|s| {
            let start = s
                .find(|c: char| c.is_alphanumeric() || c == '_')
                .unwrap_or(s.len());
            let end = s
                .rfind(|c: char| c.is_alphanumeric() || c == '_')
                .map(|i| i + 1)
                .unwrap_or(start);
            s[start..end].to_string()
        });
        !token.term.is_empty()
    }
}
