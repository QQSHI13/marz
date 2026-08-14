//! English language implementation.

use crate::language::Language;
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
            let start = s.find(|c: char| is_word_char(c)).unwrap_or(s.len());
            let end = s
                .rfind(|c: char| is_word_char(c))
                .map(|i| i + 1)
                .unwrap_or(start);
            s[start..end].to_string()
        });
        !token.term.is_empty()
    }

    fn is_stop_word(&self, term: &str) -> bool {
        STOP_WORDS.contains(&term)
    }

    fn stem(&self, term: &str) -> String {
        porter_stem(term)
    }

    fn separator_chars(&self) -> &str {
        " \t\n\r\x0C\x0B\x0D\u{00A0}-"
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

/// Porter stemmer ported from lunr.js.
fn porter_stem(w: &str) -> String {
    if w.len() < 3 {
        return w.to_string();
    }

    let mut w = w.to_string();

    // Step 1a
    if let Some(caps) = regex_captures(r"^(.+?)(ss|i)es$", &w) {
        w = format!("{}{}", caps[1], caps[2]);
    } else if let Some(caps) = regex_captures(r"^(.+?)([^s])s$", &w) {
        w = format!("{}{}", caps[1], caps[2]);
    }

    // Step 1b
    let mut step1b_done = false;
    if let Some(caps) = regex_captures(r"^(.+?)eed$", &w) {
        let stem = &caps[1];
        if measure(stem) > 0 {
            w = format!("{}e", stem);
        }
    } else if let Some(caps) = regex_captures(r"^(.+?)(ed|ing)$", &w) {
        let stem = caps[1].to_string();
        if has_vowel(&stem) {
            w = stem;
            if regex_matches(r"(at|bl|iz)$", &w) {
                w.push('e');
            } else if let Some(last) = w.chars().last() {
                if is_double_consonant(&w) && !"lsz".contains(last) {
                    w.pop();
                } else if measure(&w) == 1 && cvc_pattern(&w) {
                    w.push('e');
                }
            }
            step1b_done = true;
        }
    }

    // Step 1c
    if !step1b_done {
        if let Some(caps) = regex_captures(r"^(.+?[^aeiou])y$", &w) {
            w = format!("{}i", caps[1]);
        }
    }

    // Step 2
    let step2_replacements = [
        ("ational", "ate"),
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("bli", "ble"),
        ("alli", "al"),
        ("entli", "ent"),
        ("eli", "e"),
        ("ousli", "ous"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
        ("logi", "log"),
    ];
    for (suffix, replacement) in step2_replacements {
        if let Some(caps) = regex_captures(&format!(r"^(.+?){}$", regex::escape(suffix)), &w) {
            let stem = &caps[1];
            if measure(stem) > 0 {
                w = format!("{}{}", stem, replacement);
            }
            break;
        }
    }

    // Step 3
    let step3_replacements = [
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ful", ""),
        ("ness", ""),
    ];
    for (suffix, replacement) in step3_replacements {
        if let Some(caps) = regex_captures(&format!(r"^(.+?){}$", regex::escape(suffix)), &w) {
            let stem = &caps[1];
            if measure(stem) > 0 {
                w = format!("{}{}", stem, replacement);
            }
            break;
        }
    }

    // Step 4
    let step4_suffixes = [
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ou",
        "ism", "ate", "iti", "ous", "ive", "ize",
    ];
    let mut step4_done = false;
    for suffix in step4_suffixes {
        if let Some(caps) = regex_captures(&format!(r"^(.+?){}$", regex::escape(suffix)), &w) {
            let stem = &caps[1];
            if measure(stem) > 1 {
                w = stem.to_string();
            }
            step4_done = true;
            break;
        }
    }
    if !step4_done {
        if let Some(caps) = regex_captures(r"^(.+?)(s|t)(ion)$", &w) {
            let stem = format!("{}{}", caps[1], caps[2]);
            if measure(&stem) > 1 {
                w = stem;
            }
        }
    }

    // Step 5
    if let Some(caps) = regex_captures(r"^(.+?)e$", &w) {
        let stem = &caps[1];
        let m = measure(stem);
        if m > 1 || (m == 1 && !cvc_pattern(stem)) {
            w = stem.to_string();
        }
    }

    if w.ends_with("ll") && measure(&w) > 1 {
        w.pop();
    }

    w
}

fn regex_captures(pattern: &str, text: &str) -> Option<Vec<String>> {
    let re = regex::Regex::new(pattern).ok()?;
    let caps = re.captures(text)?;
    Some(
        caps.iter()
            .map(|m| m.map(|m| m.as_str().to_string()).unwrap_or_default())
            .collect(),
    )
}

fn regex_matches(pattern: &str, text: &str) -> bool {
    regex::Regex::new(pattern)
        .map(|re| re.is_match(text))
        .unwrap_or(false)
}

fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
}

fn has_vowel(s: &str) -> bool {
    s.chars().any(is_vowel)
}

/// Count VC sequences (the Porter "measure" m).
fn measure(s: &str) -> usize {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    if n == 0 {
        return 0;
    }

    let mut i = 0;
    // Skip initial consonants
    while i < n && !is_vowel(chars[i]) {
        i += 1;
    }

    let mut count = 0;
    while i < n {
        // Skip vowels
        while i < n && is_vowel(chars[i]) {
            i += 1;
        }
        if i >= n {
            break;
        }
        count += 1;
        // Skip consonants
        while i < n && !is_vowel(chars[i]) {
            i += 1;
        }
    }
    count
}

fn is_double_consonant(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    if n < 2 {
        return false;
    }
    let a = chars[n - 2];
    let b = chars[n - 1];
    a == b && a.is_alphabetic() && !is_vowel(a)
}

/// CVC pattern where the second C is not w, x, or y.
fn cvc_pattern(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    if n < 3 {
        return false;
    }
    let c1 = chars[n - 3];
    let v = chars[n - 2];
    let c2 = chars[n - 1];
    !is_vowel(c1) && is_vowel(v) && !is_vowel(c2) && c2 != 'w' && c2 != 'x' && c2 != 'y'
}

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
