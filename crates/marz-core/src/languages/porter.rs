//! Porter stemmer, ported from lunr.js 2.3.9.
//!
//! This is a direct port of `lunr.stemmer`, which is itself derived from the
//! classic Porter (1980) algorithm. It is written with character indices rather
//! than regular expressions so that no regex engine is needed at runtime and
//! nothing is recompiled per call.
//!
//! # Why lunr.js and not lunr.py
//!
//! The two reference implementations disagree. lunr.py implements Porter's
//! positional rule for `y` (a consonant at position 0, otherwise the opposite
//! of the preceding character), while lunr.js approximates step 1c with the
//! regex `^(.+?[^aeiou])y$`, which requires a literal non-vowel before the `y`.
//! They therefore produce different stems:
//!
//! | word | lunr.js | lunr.py |
//! |---|---|---|
//! | `deploy` | `deploy` | `deploi` |
//! | `fly` | `fli` | `fly` |
//! | `try` | `tri` | `try` |
//!
//! lunr.py's own module docstring acknowledges the divergence. Marz follows
//! **lunr.js**, because the browser is the consumer of a Marz index.
//!
//! The lunr.js regexes this port reproduces:
//!
//! ```text
//! c   = [^aeiou]              consonant
//! v   = [aeiouy]              vowel
//! C   = c[^aeiouy]*           consonant sequence
//! V   = v[aeiou]*             vowel sequence
//!
//! mgr0 = ^(C)?VC              m > 0
//! meq1 = ^(C)?VC(V)?$         m == 1
//! mgr1 = ^(C)?VCVC            m > 1
//! s_v  = ^(C)?v               contains a vowel in the stem
//! ```

/// Stem an already-lowercased English term.
///
/// Terms shorter than three characters are returned unchanged, matching lunr's
/// `if (w.length < 3) { return w }` departure from the published algorithm.
pub fn stem(word: &str) -> String {
    // The algorithm is defined over ASCII letters. Anything else (CJK, digits,
    // punctuation, accented Latin) is returned untouched rather than being
    // sliced on a byte boundary.
    if !word.is_ascii() {
        return word.to_string();
    }
    if word.len() < 3 {
        return word.to_string();
    }

    let mut w: Vec<u8> = word.as_bytes().to_vec();

    // lunr.js uppercases a leading `y` so that the `[^aeiou]` character classes
    // treat it as a consonant, then lowercases it again at the very end.
    let leading_y = w[0] == b'y';
    if leading_y {
        w[0] = b'Y';
    }

    step1a(&mut w);
    step1b(&mut w);
    step1c(&mut w);
    step2(&mut w);
    step3(&mut w);
    step4(&mut w);
    step5(&mut w);

    if leading_y && !w.is_empty() && w[0] == b'Y' {
        w[0] = b'y';
    }

    String::from_utf8(w).expect("ASCII in, ASCII out")
}

fn is_vowel_aeiou(c: u8) -> bool {
    matches!(c, b'a' | b'e' | b'i' | b'o' | b'u')
}

/// Matches lunr.js `v = [aeiouy]`.
fn is_vowel_aeiouy(c: u8) -> bool {
    matches!(c, b'a' | b'e' | b'i' | b'o' | b'u' | b'y')
}

/// Consume `C = c[^aeiouy]*` at `i`, returning the index after it.
///
/// Returns `None` when there is no consonant sequence at `i`.
fn consume_cons_seq(w: &[u8], i: usize) -> Option<usize> {
    if i >= w.len() || is_vowel_aeiou(w[i]) {
        return None;
    }
    let mut j = i + 1;
    while j < w.len() && !is_vowel_aeiouy(w[j]) {
        j += 1;
    }
    Some(j)
}

/// Consume `V = v[aeiou]*` at `i`, returning the index after it.
fn consume_vowel_seq(w: &[u8], i: usize) -> Option<usize> {
    if i >= w.len() || !is_vowel_aeiouy(w[i]) {
        return None;
    }
    let mut j = i + 1;
    while j < w.len() && is_vowel_aeiou(w[j]) {
        j += 1;
    }
    Some(j)
}

/// lunr.js `mgr0 = ^(C)?VC` — an unanchored-at-the-end prefix match.
fn measure_gt0(w: &[u8]) -> bool {
    let start = consume_cons_seq(w, 0).unwrap_or(0);
    let after_v = match consume_vowel_seq(w, start) {
        Some(j) => j,
        None => return false,
    };
    consume_cons_seq(w, after_v).is_some()
}

/// lunr.js `mgr1 = ^(C)?VCVC` — an unanchored-at-the-end prefix match.
fn measure_gt1(w: &[u8]) -> bool {
    let start = consume_cons_seq(w, 0).unwrap_or(0);
    let a = match consume_vowel_seq(w, start) {
        Some(j) => j,
        None => return false,
    };
    let b = match consume_cons_seq(w, a) {
        Some(j) => j,
        None => return false,
    };
    let c = match consume_vowel_seq(w, b) {
        Some(j) => j,
        None => return false,
    };
    consume_cons_seq(w, c).is_some()
}

/// lunr.js `meq1 = ^(C)?VC(V)?$` — fully anchored.
fn measure_eq1(w: &[u8]) -> bool {
    let start = consume_cons_seq(w, 0).unwrap_or(0);
    let a = match consume_vowel_seq(w, start) {
        Some(j) => j,
        None => return false,
    };
    let b = match consume_cons_seq(w, a) {
        Some(j) => j,
        None => return false,
    };
    let end = consume_vowel_seq(w, b).unwrap_or(b);
    end == w.len()
}

/// lunr.js `s_v = ^(C)?v` — is there a vowel after any leading consonant run.
fn has_vowel_in_stem(w: &[u8]) -> bool {
    let start = consume_cons_seq(w, 0).unwrap_or(0);
    start < w.len() && is_vowel_aeiouy(w[start])
}

/// lunr.js `re3_1b_2 = ([^aeiouylsz])\1$` — a doubled consonant, excluding
/// `l`, `s`, `z` (and vowels and `y`).
fn ends_double_consonant(w: &[u8]) -> bool {
    let n = w.len();
    if n < 2 {
        return false;
    }
    let a = w[n - 2];
    let b = w[n - 1];
    a == b
        && !matches!(
            a,
            b'a' | b'e' | b'i' | b'o' | b'u' | b'y' | b'l' | b's' | b'z'
        )
}

/// lunr.js `^C v [^aeiouwxy]$` — the "restore a final e" CVC test.
///
/// Note this is anchored at both ends, so it applies to the whole slice.
fn is_cvc(w: &[u8]) -> bool {
    let after_c = match consume_cons_seq(w, 0) {
        Some(j) => j,
        None => return false,
    };
    // Need exactly v then a final consonant that is not w, x or y.
    if after_c + 2 != w.len() {
        return false;
    }
    if !is_vowel_aeiouy(w[after_c]) {
        return false;
    }
    let last = w[after_c + 1];
    !is_vowel_aeiouy(last) && !matches!(last, b'w' | b'x' | b'y')
}

fn ends_with(w: &[u8], suffix: &[u8]) -> bool {
    w.len() >= suffix.len() && &w[w.len() - suffix.len()..] == suffix
}

/// Step 1a: plurals.
///
/// lunr.js:
/// ```text
/// re_1a  = /^(.+?)(ss|i)es$/  -> $1$2
/// re2_1a = /^(.+?)([^s])s$/   -> $1$2
/// ```
/// Because `.+?` is lazy but the suffix is anchored, these reduce to simple
/// suffix tests. The `.+?` requires at least one character before the suffix.
fn step1a(w: &mut Vec<u8>) {
    if ends_with(w, b"sses") && w.len() > 4 {
        // (.+?)(ss)es -> $1ss
        w.truncate(w.len() - 2);
    } else if ends_with(w, b"ies") && w.len() > 3 {
        // (.+?)(i)es -> $1i
        w.truncate(w.len() - 2);
    } else if ends_with(w, b"s") && w.len() > 2 {
        // (.+?)([^s])s -> $1$2 : drop the final s unless preceded by s
        let n = w.len();
        if w[n - 2] != b's' {
            w.truncate(n - 1);
        }
    }
}

/// Step 1b: `-eed`, `-ed`, `-ing`.
///
/// lunr.js:
/// ```text
/// re_1b  = /^(.+?)eed$/
/// re2_1b = /^(.+?)(ed|ing)$/
/// ```
/// For the `eed` branch lunr.js tests `mgr0` on the captured stem and then does
/// `w = w.replace(/.$/, "")` — that strips exactly one character from the
/// **whole word**, not from the captured stem. So `agreed` -> `agree`, not
/// `agr` + `e`.
fn step1b(w: &mut Vec<u8>) {
    if ends_with(w, b"eed") && w.len() > 3 {
        let stem = &w[..w.len() - 3];
        if measure_gt0(stem) {
            w.truncate(w.len() - 1);
        }
        return;
    }

    let suffix_len = if ends_with(w, b"ed") && w.len() > 2 {
        2
    } else if ends_with(w, b"ing") && w.len() > 3 {
        3
    } else {
        return;
    };

    let stem_len = w.len() - suffix_len;
    if !has_vowel_in_stem(&w[..stem_len]) {
        return;
    }
    w.truncate(stem_len);

    if ends_with(w, b"at") || ends_with(w, b"bl") || ends_with(w, b"iz") {
        w.push(b'e');
    } else if ends_double_consonant(w) {
        w.truncate(w.len() - 1);
    } else if is_cvc(w) {
        w.push(b'e');
    }
}

/// Step 1c: terminal `y` -> `i`.
///
/// lunr.js `re_1c = /^(.+?[^aeiou])y$/`, applied **unconditionally** after
/// step 1b (not only when step 1b failed to fire). This is what turns
/// `semidrying` -> `semidry` -> `semidri`.
///
/// The `[^aeiou]` class excludes only `aeiou`, so an uppercased leading `Y`
/// counts as a non-vowel here, matching lunr.js.
fn step1c(w: &mut [u8]) {
    let n = w.len();
    // Need `.+?` (>= 1 char) then a non-vowel, then the final y: so >= 3 chars.
    if n < 3 || w[n - 1] != b'y' {
        return;
    }
    if is_vowel_aeiou(w[n - 2]) {
        return;
    }
    w[n - 1] = b'i';
}

/// Step 2 suffix map, ordered longest-first within each shared ending.
///
/// lunr.js uses one alternation regex with a lazy `.+?`, so the engine prefers
/// the match that leaves the **shortest stem**, i.e. the longest suffix. A
/// naive first-match-in-list scan is wrong: `dismayable` must match `able`
/// (step 4), not `al`. Sorting by descending length reproduces the regex
/// behaviour.
const STEP2: &[(&[u8], &[u8])] = &[
    (b"ational", b"ate"),
    (b"fulness", b"ful"),
    (b"iveness", b"ive"),
    (b"ousness", b"ous"),
    (b"ization", b"ize"),
    (b"tional", b"tion"),
    (b"biliti", b"ble"),
    (b"lizer", b"lize"),
    (b"alism", b"al"),
    (b"ation", b"ate"),
    (b"entli", b"ent"),
    (b"ousli", b"ous"),
    (b"aliti", b"al"),
    (b"iviti", b"ive"),
    (b"anci", b"ance"),
    (b"enci", b"ence"),
    (b"izer", b"ize"),
    (b"alli", b"al"),
    (b"ator", b"ate"),
    (b"logi", b"log"),
    (b"bli", b"ble"),
    (b"eli", b"e"),
];

/// Step 3 suffix map, longest-first.
const STEP3: &[(&[u8], &[u8])] = &[
    (b"icate", b"ic"),
    (b"ative", b""),
    (b"alize", b"al"),
    (b"iciti", b"ic"),
    (b"ical", b"ic"),
    (b"ness", b""),
    (b"ful", b""),
];

/// Step 4 suffixes, longest-first.
const STEP4: &[&[u8]] = &[
    b"ement", b"ance", b"ence", b"able", b"ible", b"ment", b"ant", b"ent", b"ism", b"ate", b"iti",
    b"ous", b"ive", b"ize", b"al", b"er", b"ic", b"ou",
];

/// Apply the first matching `(suffix, replacement)` pair whose stem has m > 0.
///
/// lunr.js tests the alternation first and only then checks `mgr0`; if the
/// stem fails the measure test the word is left unchanged and **no other**
/// suffix is tried. The `break`-on-match (not on-success) behaviour matters.
fn apply_suffix_map(w: &mut Vec<u8>, table: &[(&[u8], &[u8])]) {
    for (suffix, replacement) in table {
        // `.+?` requires at least one character before the suffix.
        if w.len() <= suffix.len() || !ends_with(w, suffix) {
            continue;
        }
        let stem_len = w.len() - suffix.len();
        if measure_gt0(&w[..stem_len]) {
            w.truncate(stem_len);
            w.extend_from_slice(replacement);
        }
        return;
    }
}

fn step2(w: &mut Vec<u8>) {
    apply_suffix_map(w, STEP2);
}

fn step3(w: &mut Vec<u8>) {
    apply_suffix_map(w, STEP3);
}

/// Step 4: strip a suffix when the remaining stem has m > 1.
///
/// lunr.js tries `re_4` first, and only if that alternation does not match at
/// all does it try `re2_4 = /^(.+?)(s|t)(ion)$/`.
fn step4(w: &mut Vec<u8>) {
    for suffix in STEP4 {
        if w.len() <= suffix.len() || !ends_with(w, suffix) {
            continue;
        }
        let stem_len = w.len() - suffix.len();
        if measure_gt1(&w[..stem_len]) {
            w.truncate(stem_len);
        }
        return;
    }

    // (.+?)(s|t)(ion) -> stem is $1$2, i.e. everything except "ion".
    if ends_with(w, b"ion") && w.len() > 4 {
        let stem_len = w.len() - 3;
        if matches!(w[stem_len - 1], b's' | b't') && measure_gt1(&w[..stem_len]) {
            w.truncate(stem_len);
        }
    }
}

/// Step 5: drop a final `e`, and reduce `ll` to `l`.
fn step5(w: &mut Vec<u8>) {
    if ends_with(w, b"e") && w.len() > 1 {
        let stem_len = w.len() - 1;
        let stem = &w[..stem_len];
        if measure_gt1(stem) || (measure_eq1(stem) && !is_cvc(stem)) {
            w.truncate(stem_len);
        }
    }

    if ends_with(w, b"ll") && measure_gt1(w) {
        w.truncate(w.len() - 1);
    }
}

#[cfg(test)]
mod tests {
    use super::stem;

    /// Cases verified against lunr.js 2.3.9 via node.
    #[test]
    fn matches_lunr_js() {
        let cases = [
            // Regression cases for bugs in the previous regex implementation.
            ("agreed", "agre"),        // eed strips one char from the whole word
            ("semidrying", "semidri"), // step 1c runs after step 1b fires
            ("dismayable", "dismay"),  // longest suffix wins, not first in list
            ("blisterweed", "blisterwe"),
            ("pepperweed", "pepperwe"),
            ("tallowweed", "tallowwe"),
            // lunr.js-specific y handling (lunr.py differs on all of these).
            ("deploy", "deploy"),
            ("deployed", "deploy"),
            ("enjoy", "enjoy"),
            ("journey", "journey"),
            ("fly", "fli"),
            ("try", "tri"),
            ("sky", "ski"),
            ("dry", "dri"),
            ("lay", "lay"),
            ("say", "say"),
            ("buy", "buy"),
            ("boy", "boy"),
            ("key", "key"),
            ("toy", "toy"),
            ("gray", "gray"),
            ("play", "play"),
            ("they", "they"),
            ("money", "money"),
            ("valley", "valley"),
            ("yellow", "yellow"),
            // Classic Porter cases.
            ("caress", "caress"),
            ("cats", "cat"),
            ("ponies", "poni"),
            ("ties", "ti"),
            ("feed", "feed"),
            ("plastered", "plaster"),
            ("motoring", "motor"),
            ("sing", "sing"),
            ("conflated", "conflat"),
            ("conflate", "conflat"),
            ("troubled", "troubl"),
            ("sized", "size"),
            ("hopping", "hop"),
            ("falling", "fall"),
            ("probate", "probat"),
            ("rates", "rate"),
            ("controlled", "control"),
            ("rolling", "roll"),
            ("abilities", "abil"),
            ("relational", "relat"),
            ("markdown", "markdown"),
            ("running", "run"),
            ("flies", "fli"),
            ("happy", "happi"),
            ("study", "studi"),
            ("national", "nation"),
            ("died", "di"),
        ];
        for (input, expected) in cases {
            assert_eq!(stem(input), expected, "stem({input:?})");
        }
    }

    #[test]
    fn short_words_pass_through() {
        for w in ["a", "be", "is", "go"] {
            assert_eq!(stem(w), w);
        }
    }

    #[test]
    fn non_ascii_passes_through_without_panicking() {
        // Must not slice a multi-byte character.
        for w in ["中文", "café", "naïve", "日本語", "한국어", "«é»"] {
            assert_eq!(stem(w), w);
        }
    }
}
