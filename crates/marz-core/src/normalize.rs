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
fn halfwidth_katakana(c: char) -> Option<char> {
    // U+FF61..U+FF9F is the half-width katakana block (plus punctuation).
    let table: &[(char, char)] = &[
        ('\u{FF61}', '\u{3002}'), // 。
        ('\u{FF62}', '\u{300C}'), // 「
        ('\u{FF63}', '\u{300D}'), // 」
        ('\u{FF64}', '\u{3001}'), // 、
        ('\u{FF65}', '\u{30FB}'), // ・
        ('\u{FF66}', '\u{30F2}'), // ヲ
        ('\u{FF67}', '\u{30A1}'), // ァ
        ('\u{FF68}', '\u{30A3}'), // ィ
        ('\u{FF69}', '\u{30A5}'), // ゥ
        ('\u{FF6A}', '\u{30A7}'), // ェ
        ('\u{FF6B}', '\u{30A9}'), // ォ
        ('\u{FF6C}', '\u{30E3}'), // ャ
        ('\u{FF6D}', '\u{30E5}'), // ュ
        ('\u{FF6E}', '\u{30E7}'), // ョ
        ('\u{FF6F}', '\u{30C3}'), // ッ
        ('\u{FF70}', '\u{30FC}'), // ー
        ('\u{FF71}', '\u{30A2}'), // ア
        ('\u{FF72}', '\u{30A4}'), // イ
        ('\u{FF73}', '\u{30A6}'), // ウ
        ('\u{FF74}', '\u{30A8}'), // エ
        ('\u{FF75}', '\u{30AA}'), // オ
        ('\u{FF76}', '\u{30AB}'), // カ
        ('\u{FF77}', '\u{30AD}'), // キ
        ('\u{FF78}', '\u{30AF}'), // ク
        ('\u{FF79}', '\u{30B1}'), // ケ
        ('\u{FF7A}', '\u{30B3}'), // コ
        ('\u{FF7B}', '\u{30B5}'), // サ
        ('\u{FF7C}', '\u{30B7}'), // シ
        ('\u{FF7D}', '\u{30B9}'), // ス
        ('\u{FF7E}', '\u{30BB}'), // セ
        ('\u{FF7F}', '\u{30BD}'), // ソ
        ('\u{FF80}', '\u{30BF}'), // タ
        ('\u{FF81}', '\u{30C1}'), // チ
        ('\u{FF82}', '\u{30C4}'), // ツ
        ('\u{FF83}', '\u{30C6}'), // テ
        ('\u{FF84}', '\u{30C8}'), // ト
        ('\u{FF85}', '\u{30CA}'), // ナ
        ('\u{FF86}', '\u{30CB}'), // ニ
        ('\u{FF87}', '\u{30CC}'), // ヌ
        ('\u{FF88}', '\u{30CD}'), // ネ
        ('\u{FF89}', '\u{30CE}'), // ノ
        ('\u{FF8A}', '\u{30CF}'), // ハ
        ('\u{FF8B}', '\u{30D2}'), // ヒ
        ('\u{FF8C}', '\u{30D5}'), // フ
        ('\u{FF8D}', '\u{30D8}'), // ヘ
        ('\u{FF8E}', '\u{30DB}'), // ホ
        ('\u{FF8F}', '\u{30DE}'), // マ
        ('\u{FF90}', '\u{30DF}'), // ミ
        ('\u{FF91}', '\u{30E0}'), // ム
        ('\u{FF92}', '\u{30E1}'), // メ
        ('\u{FF93}', '\u{30E2}'), // モ
        ('\u{FF94}', '\u{30E4}'), // ヤ
        ('\u{FF95}', '\u{30E6}'), // ユ
        ('\u{FF96}', '\u{30E8}'), // ヨ
        ('\u{FF97}', '\u{30E9}'), // ラ
        ('\u{FF98}', '\u{30EA}'), // リ
        ('\u{FF99}', '\u{30EB}'), // ル
        ('\u{FF9A}', '\u{30EC}'), // レ
        ('\u{FF9B}', '\u{30ED}'), // ロ
        ('\u{FF9C}', '\u{30EF}'), // ワ
        ('\u{FF9D}', '\u{30F3}'), // ン
    ];
    table
        .iter()
        .find(|(half, _)| *half == c)
        .map(|(_, full)| *full)
}

/// Compose a katakana base with the voiced sound mark, e.g. カ + ﾞ -> ガ.
///
/// In the Unicode katakana block the voiced form is almost always the base
/// code point plus one; the `ウ -> ヴ` case is the exception.
fn compose_voiced(base: char) -> Option<char> {
    const VOICEABLE: &str = "カキクケコサシスセソタチツテトハヒフヘホ";
    if VOICEABLE.contains(base) {
        return char::from_u32(base as u32 + 1);
    }
    if base == 'ウ' {
        return Some('\u{30F4}'); // ヴ
    }
    None
}

/// Compose a katakana base with the semi-voiced sound mark, e.g. ハ + ﾟ -> パ.
fn compose_semi_voiced(base: char) -> Option<char> {
    const SEMI_VOICEABLE: &str = "ハヒフヘホ";
    if SEMI_VOICEABLE.contains(base) {
        return char::from_u32(base as u32 + 2);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::normalize;

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
