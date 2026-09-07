//! Dictionary-free, dependency-free text normalization for CJK and Latin text.
//!
//! Real-world CJK documents mix several width variants of the same character:
//! full-width Latin (`ＲＵＳＴ`), full-width digits (`２０２４`), half-width
//! katakana (`ｶﾞｲﾄﾞ`), and the ideographic space (`　`). A searcher who types
//! `rust` should match `ＲＵＳＴ`, and one who types `ガイド` should match
//! `ｶﾞｲﾄﾞ`. Without folding, those are simply different terms and the search
//! silently fails.
//!
//! This is the subset of Unicode NFKC that matters for search. It is a handful
//! of arithmetic ranges plus one small katakana table — no `unicode-*` crate,
//! no data files.

/// Normalize text for indexing and querying.
///
/// Applies, in order:
/// 1. Full-width ASCII (U+FF01–U+FF5E) → ASCII.
/// 2. Ideographic and other exotic spaces → U+0020.
/// 3. Half-width katakana → full-width katakana, combining the voiced sound
///    marks `ﾞ` and `ﾟ` into the preceding character where a composed form
///    exists.
/// 4. Lowercasing.
///
/// Both the indexer and the query parser must call this, or the two sides will
/// disagree.
pub fn normalize(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        // Half-width katakana, possibly followed by a voiced sound mark.
        if let Some(base) = halfwidth_katakana(c) {
            let next = chars.get(i + 1).copied();
            match next {
                Some('\u{FF9E}') => {
                    // Voiced: ｶ + ﾞ -> ガ
                    if let Some(voiced) = compose_voiced(base) {
                        out.push(voiced);
                        i += 2;
                        continue;
                    }
                }
                Some('\u{FF9F}') => {
                    // Semi-voiced: ﾊ + ﾟ -> パ
                    if let Some(semi) = compose_semi_voiced(base) {
                        out.push(semi);
                        i += 2;
                        continue;
                    }
                }
                _ => {}
            }
            out.push(base);
            i += 1;
            continue;
        }

        out.extend(fold_char(c));
        i += 1;
    }

    out.to_lowercase()
}

/// Normalize with the lowercasing rules of `code`.
///
/// Today only Turkish differs (`tr` needs dotless/dotted-I handling); every
/// other code folds through [`normalize`]. Both the indexer and the query
/// parser must use this same dispatch, or the two sides disagree on what
/// `Istanbul` becomes and Turkish queries silently miss.
pub fn normalize_for_language(code: &str, text: &str) -> String {
    if code == "tr" {
        normalize_tr(text)
    } else {
        normalize(text)
    }
}

/// Normalize with Turkish dotless/dotted-I rules.
///
/// Identical to [`normalize`] except the final lowercasing: `I` → `ı`
/// (U+0131) and `İ` → `i`, matching Turkish orthography. Unicode default
/// lowercasing maps `I` → `i`, which merges two distinct Turkish letters and
/// feeds the Snowball Turkish stemmer input it was not designed for. Both the
/// indexer and the query path must use the same variant, which
/// `SnowballLanguage` ensures by dispatching on its code.
pub fn normalize_tr(text: &str) -> String {
    // Reuse the width-folding half, then Turkish-lowercase.
    let folded = normalize_fold_only(text);
    let mut out = String::with_capacity(folded.len());
    for c in folded.chars() {
        match c {
            'I' => out.push('ı'),
            'İ' => out.push('i'),
            _ => out.extend(c.to_lowercase()),
        }
    }
    out
}

/// Width-folding half of [`normalize`] without lowercasing, shared by both
/// lowercasing variants.
fn normalize_fold_only(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if let Some(base) = halfwidth_katakana(c) {
            let next = chars.get(i + 1).copied();
            match next {
                Some('\u{FF9E}') => {
                    if let Some(voiced) = compose_voiced(base) {
                        out.push(voiced);
                        i += 2;
                        continue;
                    }
                }
                Some('\u{FF9F}') => {
                    if let Some(semi) = compose_semi_voiced(base) {
                        out.push(semi);
                        i += 2;
                        continue;
                    }
                }
                _ => {}
            }
            out.push(base);
            i += 1;
            continue;
        }
        out.extend(fold_char(c));
        i += 1;
    }
    out
}

/// Fold a single character that needs no lookahead.
fn fold_char(c: char) -> impl Iterator<Item = char> {
    let folded = match c as u32 {
        // Full-width ASCII variants -> ASCII.
        0xFF01..=0xFF5E => char::from_u32(c as u32 - 0xFF01 + 0x21).unwrap_or(c),
        // Ideographic space and assorted Unicode spaces -> plain space.
        0x3000 | 0x2000..=0x200A | 0x202F | 0x205F => ' ',
        // Full-width / wave / fullwidth macron oddities that appear in the wild.
        0xFFE5 => '¥',
        _ => c,
    };
    std::iter::once(folded)
}

/// Map a half-width katakana code point to its full-width base form.
///
/// O(1) table lookup indexed by `c - 0xFF61`. `None` for the voiced/semi-voiced
/// marks themselves (`FF9E`/`FF9F`) and anything outside the block.
fn halfwidth_katakana(c: char) -> Option<char> {
    // U+FF61..=U+FF9D in order (FF9E/FF9F are the sound marks, handled by the
    // caller). `'\0'` is unused — the block is dense, so every slot maps.
    const TABLE: &[char] = &[
        '\u{3002}', // FF61 。
        '\u{300C}', // FF62 「
        '\u{300D}', // FF63 」
        '\u{3001}', // FF64 、
        '\u{30FB}', // FF65 ・
        '\u{30F2}', // FF66 ヲ
        '\u{30A1}', // FF67 ァ
        '\u{30A3}', // FF68 ィ
        '\u{30A5}', // FF69 ゥ
        '\u{30A7}', // FF6A ェ
        '\u{30A9}', // FF6B ォ
        '\u{30E3}', // FF6C ャ
        '\u{30E5}', // FF6D ュ
        '\u{30E7}', // FF6E ョ
        '\u{30C3}', // FF6F ッ
        '\u{30FC}', // FF70 ー
        '\u{30A2}', // FF71 ア
        '\u{30A4}', // FF72 イ
        '\u{30A6}', // FF73 ウ
        '\u{30A8}', // FF74 エ
        '\u{30AA}', // FF75 オ
        '\u{30AB}', // FF76 カ
        '\u{30AD}', // FF77 キ
        '\u{30AF}', // FF78 ク
        '\u{30B1}', // FF79 ケ
        '\u{30B3}', // FF7A コ
        '\u{30B5}', // FF7B サ
        '\u{30B7}', // FF7C シ
        '\u{30B9}', // FF7D ス
        '\u{30BB}', // FF7E セ
        '\u{30BD}', // FF7F ソ
        '\u{30BF}', // FF80 タ
        '\u{30C1}', // FF81 チ
        '\u{30C4}', // FF82 ツ
        '\u{30C6}', // FF83 テ
        '\u{30C8}', // FF84 ト
        '\u{30CA}', // FF85 ナ
        '\u{30CB}', // FF86 ニ
        '\u{30CC}', // FF87 ヌ
        '\u{30CD}', // FF88 ネ
        '\u{30CE}', // FF89 ノ
        '\u{30CF}', // FF8A ハ
        '\u{30D2}', // FF8B ヒ
        '\u{30D5}', // FF8C フ
        '\u{30D8}', // FF8D ヘ
        '\u{30DB}', // FF8E ホ
        '\u{30DE}', // FF8F マ
        '\u{30DF}', // FF90 ミ
        '\u{30E0}', // FF91 ム
        '\u{30E1}', // FF92 メ
        '\u{30E2}', // FF93 モ
        '\u{30E4}', // FF94 ヤ
        '\u{30E6}', // FF95 ユ
        '\u{30E8}', // FF96 ヨ
        '\u{30E9}', // FF97 ラ
        '\u{30EA}', // FF98 リ
        '\u{30EB}', // FF99 ル
        '\u{30EC}', // FF9A レ
        '\u{30ED}', // FF9B ロ
        '\u{30EF}', // FF9C ワ
        '\u{30F3}', // FF9D ン
    ];
    let idx = (c as u32).wrapping_sub(0xFF61) as usize;
    TABLE.get(idx).copied()
}

/// Compose a katakana base with the voiced sound mark, e.g. カ + ﾞ -> ガ.
///
/// In the Unicode katakana block the voiced form is almost always the base
/// code point plus one; the `ウ -> ヴ`, `ワ -> ヷ`, `ヲ -> ヺ` cases are the
/// exceptions.
fn compose_voiced(base: char) -> Option<char> {
    if matches!(
        base,
        'カ' | 'キ'
            | 'ク'
            | 'ケ'
            | 'コ'
            | 'サ'
            | 'シ'
            | 'ス'
            | 'セ'
            | 'ソ'
            | 'タ'
            | 'チ'
            | 'ツ'
            | 'テ'
            | 'ト'
            | 'ハ'
            | 'ヒ'
            | 'フ'
            | 'ヘ'
            | 'ホ'
    ) {
        return char::from_u32(base as u32 + 1);
    }
    match base {
        'ウ' => Some('\u{30F4}'), // ヴ
        'ワ' => Some('\u{30F7}'), // ヷ
        'ヲ' => Some('\u{30FA}'), // ヺ
        _ => None,
    }
}

/// Compose a katakana base with the semi-voiced sound mark, e.g. ハ + ﾟ -> パ.
fn compose_semi_voiced(base: char) -> Option<char> {
    if matches!(base, 'ハ' | 'ヒ' | 'フ' | 'ヘ' | 'ホ') {
        return char::from_u32(base as u32 + 2);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{normalize, normalize_for_language};

    #[test]
    fn turkish_folds_dotted_and_dotless_i() {
        assert_eq!(normalize_for_language("tr", "Istanbul"), "ıstanbul");
        assert_eq!(normalize_for_language("tr", "İZMİR"), "izmir");
        // Every other language uses the default fold.
        assert_eq!(normalize_for_language("en", "Istanbul"), "istanbul");
        assert_eq!(normalize_for_language("de", "Istanbul"), "istanbul");
    }

    #[test]
    fn folds_fullwidth_ascii() {
        assert_eq!(normalize("ＲＵＳＴ"), "rust");
        assert_eq!(normalize("２０２４"), "2024");
        assert_eq!(normalize("ｈｅｌｌｏ！"), "hello!");
    }

    #[test]
    fn folds_ideographic_space() {
        assert_eq!(normalize("中文　搜索"), "中文 搜索");
    }

    #[test]
    fn folds_halfwidth_katakana() {
        assert_eq!(normalize("ｶﾞｲﾄﾞ"), "ガイド");
        assert_eq!(normalize("ﾊﾟﾝ"), "パン");
        assert_eq!(normalize("ｼﾝｸﾞﾙ"), "シングル");
        assert_eq!(normalize("ｳﾞ"), "ヴ");
    }

    #[test]
    fn lowercases_latin() {
        assert_eq!(normalize("Hello World"), "hello world");
    }

    #[test]
    fn leaves_cjk_alone() {
        assert_eq!(normalize("中文搜索引擎"), "中文搜索引擎");
        assert_eq!(normalize("검색 엔진"), "검색 엔진");
    }

    #[test]
    fn lone_sound_mark_does_not_panic() {
        // A voiced mark with no composable base must pass through.
        assert_eq!(normalize("\u{FF9E}"), "\u{FF9E}");
        assert_eq!(normalize("ｱ\u{FF9E}"), "ア\u{FF9E}");
    }
}
