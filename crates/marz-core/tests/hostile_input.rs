//! Inputs that a search box will eventually receive, and must survive.
//!
//! Two sources of hostile input, with different consequences.
//!
//! **Queries** come from whoever is typing. A query that panics takes down the
//! page in the browser and the process in Python — a search box is the one place
//! in an application where arbitrary text reaches a parser, so every malformed
//! query must return an error or empty results, never unwind. The tests here
//! therefore assert *no panic* far more often than they assert a specific
//! result: the specific results are `query_parser`'s unit tests, while this file
//! is about the boundary.
//!
//! **Index bytes** come from a network fetch. Truncation is covered in
//! `binary_roundtrip.rs`; this file covers corruption, which is a different
//! failure mode. A truncated index has a consistent prefix and simply ends,
//! while a single flipped byte in a section offset points the reader at
//! arbitrary data that still passes a length check. That is the case where a
//! reader built on unchecked slicing reads out of bounds instead of failing.

use std::sync::Arc;

use marz_core::languages::{Chinese, English, Japanese, Korean};
use marz_core::{Index, IndexBuilder, Language};

fn english_index() -> Index {
    let mut builder = IndexBuilder::new(Arc::new(English) as Arc<dyn Language>);
    builder
        .ref_field("id")
        .field("title", 10.0)
        .field("body", 1.0);
    builder.add("a", 1.0, |name| match name {
        "title" => Some("Keyboards".to_string()),
        "body" => Some("A keyboard has keys and a cable.".to_string()),
        _ => None,
    });
    builder.add("b", 1.0, |name| match name {
        "title" => Some("Mice".to_string()),
        "body" => Some("A mouse has no keys at all.".to_string()),
        _ => None,
    });
    builder.build()
}

fn japanese_index() -> Index {
    let mut builder = IndexBuilder::new(Arc::new(Japanese) as Arc<dyn Language>);
    builder.ref_field("id").field("body", 1.0);
    builder.add("ja", 1.0, |name| match name {
        "body" => Some("検索エンジンの仕組みと機械学習について。".to_string()),
        _ => None,
    });
    builder.build()
}

/// A larger index, so byte-corruption sweeps reach the postings and positions
/// sections rather than spending most of their trials inside the header and the
/// term dictionary.
fn corruptible_index() -> Index {
    const DOCS: &[(&str, &str)] = &[
        (
            "ja/1",
            "検索エンジンの仕組みについて説明します。全文検索は索引を使います。",
        ),
        (
            "ja/2",
            "検索の話とエンジンオイルとジンの話をします。機械のエンジンです。",
        ),
        (
            "ja/3",
            "機械学習は人工知能の一分野です。統計的な学習理論に基づきます。",
        ),
        (
            "ja/4",
            "形態素解析を使わずに検索する方法を考えます。辞書は不要です。",
        ),
        (
            "ja/5",
            "日本語の検索エンジンをブラウザで動かす。ｶﾞｲﾄﾞと々を含む。",
        ),
    ];

    let mut builder = IndexBuilder::new(Arc::new(Japanese) as Arc<dyn Language>);
    builder
        .ref_field("id")
        .field("title", 10.0)
        .field("body", 1.0);
    for (id, body) in DOCS {
        let body = body.to_string();
        builder.add(*id, 1.0, move |name| match name {
            "title" => Some(body.chars().take(6).collect()),
            "body" => Some(body.clone()),
            _ => None,
        });
    }
    builder.build()
}

/// Queries that are syntactically odd, empty, or degenerate.
///
/// Each must produce a result or an error. The assertion is that control returns
/// at all: `search` returning `Err` is a fine outcome for any of these, and so is
/// an empty result set. Panicking is not.
const HOSTILE_QUERIES: &[&str] = &[
    // Empty and whitespace.
    "",
    " ",
    "\t\n",
    "\u{3000}", // ideographic space
    "\u{00a0}", // non-breaking space
    "\u{200b}", // zero-width space: normalizes to nothing
    // Operators with no operand.
    "+",
    "-",
    "*",
    "~",
    "^",
    ":",
    "+-",
    "^^",
    "~~",
    "**",
    ":::",
    // Operators attached to nothing usable.
    "+*",
    "-*",
    "keyboard^",
    "keyboard~",
    "keyboard:",
    ":keyboard",
    "title:",
    "^5",
    "~1",
    // Numeric arguments that are absent, huge, negative or not numbers.
    "keyboard^0",
    "keyboard^-1",
    "keyboard^999999999999999999999",
    "keyboard^1e400",
    "keyboard^abc",
    "keyboard~0",
    "keyboard~-1",
    "keyboard~99",
    "keyboard~999999999999999999999",
    "keyboard~abc",
    // Wildcards that match everything, or nothing, or are pathological.
    "*",
    "**",
    "***",
    "*a*",
    "a*b*c*d*e*f*g*",
    // Field scoping gone wrong.
    "title:body:keyboard",
    "title:*",
    "nosuchfield:keyboard", // legitimately an error
    // Combinations of modifiers on one term.
    "+title:keyboard^10~1",
    "-title:keyboard^10~1",
    "+keyboard -keyboard",
    "+nothingmatches +keyboard",
    // Very long input. A search box with a paste in it.
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa*",
    // Text that is only punctuation, which the trimmer removes entirely.
    "...",
    "---",
    "!!!",
    "()",
    "\"\"",
    "''",
    // Mixed scripts and stray marks.
    "検索 keyboard 검색 中文",
    "\u{309b}", // lone voiced sound mark
    "\u{ff9e}", // lone halfwidth voiced sound mark
    "\u{fe0f}", // lone variation selector
    "🎉",       // astral plane only
    "🎉*",
    "e\u{0301}", // combining acute, decomposed
    // Control characters, which arrive from bad copy-paste.
    "\u{0000}",
    "keyboard\u{0000}",
    "\u{001b}[31m",
    // Right-to-left and bidi overrides.
    "\u{202e}keyboard",
    "مرحبا",
    // Surrogate-adjacent and unusual planes.
    "\u{10ffff}",
    "\u{e0001}",
];

#[test]
fn no_query_panics_the_english_index() {
    let index = english_index();
    for query in HOSTILE_QUERIES {
        // Deliberately discarding: an Err is a valid outcome for a malformed
        // query, and asserting which of these parse would pin behaviour this
        // test is not about.
        let _ = index.search(query);
    }
}

#[test]
fn no_query_panics_the_japanese_index() {
    // The CJK path differs enough to be worth running separately: bigram
    // tokenization can turn a one-character query into zero terms, and phrase
    // verification reads positions that a degenerate query may not have.
    let index = japanese_index();
    for query in HOSTILE_QUERIES {
        let _ = index.search(query);
    }
}

#[test]
fn no_query_panics_any_language() {
    // Chinese and Korean tokenize differently again — Korean on whitespace,
    // Chinese in bigrams with no spaces to fall back on.
    for language in [
        Arc::new(Chinese) as Arc<dyn Language>,
        Arc::new(Korean) as Arc<dyn Language>,
    ] {
        let mut builder = IndexBuilder::new(language);
        builder.ref_field("id").field("body", 1.0);
        builder.add("x", 1.0, |name| match name {
            "body" => Some("검색 엔진 中文搜索 mixed text".to_string()),
            _ => None,
        });
        let index = builder.build();
        for query in HOSTILE_QUERIES {
            let _ = index.search(query);
        }
    }
}

#[test]
fn an_empty_query_returns_nothing_rather_than_everything() {
    // The failure this guards is specific and user-visible: a search box that
    // lists the entire corpus the moment it is focused.
    let index = english_index();
    for query in ["", " ", "\t", "\u{3000}", "\u{200b}"] {
        let results = index.search(query).unwrap_or_default();
        assert!(
            results.is_empty(),
            "query {query:?} returned {} results; an empty query must match nothing",
            results.len()
        );
    }
}

#[test]
fn a_punctuation_only_query_returns_nothing() {
    // These tokenize to terms the trimmer empties. A term that trims to nothing
    // must be dropped, not matched as the empty string — which would match
    // every document.
    let index = english_index();
    for query in ["...", "---", "!!!", "()", "\u{309b}"] {
        let results = index.search(query).unwrap_or_default();
        assert!(
            results.is_empty(),
            "query {query:?} returned {} results",
            results.len()
        );
    }
}

#[test]
fn documents_whose_fields_are_hostile_still_index() {
    // The other direction: the corpus is the hostile input. A build step feeds
    // whatever the content authors wrote, including text that trims to nothing.
    let hostile: &[&str] = &[
        "",
        " ",
        "\u{200b}",
        "...",
        "\u{0000}",
        "\u{309b}\u{309b}\u{309b}",
        "🎉🎉🎉",
        "\u{202e}reversed",
        "e\u{0301}cole",
        "ｶﾞｲﾄﾞ",
        "々々々",
        "a\u{0000}b",
    ];

    let mut builder = IndexBuilder::new(Arc::new(Japanese) as Arc<dyn Language>);
    builder.ref_field("id").field("body", 1.0);
    for (i, text) in hostile.iter().enumerate() {
        let text = text.to_string();
        let id = format!("doc-{i}");
        builder.add(id, 1.0, move |name| match name {
            "body" => Some(text.clone()),
            _ => None,
        });
    }
    let index = builder.build();

    // Every document is present regardless of whether it produced any terms:
    // dropping a document because its text trimmed away would silently shrink
    // the corpus.
    assert_eq!(index.document_count(), hostile.len());

    // And the index answers queries, including for the text that survived
    // normalization.
    assert!(!index.search("ガイド").unwrap().is_empty());
    let _ = index.search("reversed");
    let _ = index.search("école");
}

#[test]
fn a_field_of_only_separators_does_not_break_average_length() {
    // Average field length is a BM25 divisor. A corpus where some fields are
    // empty must not produce a zero divisor, an infinity, or a NaN score.
    let mut builder = IndexBuilder::new(Arc::new(English) as Arc<dyn Language>);
    builder
        .ref_field("id")
        .field("title", 1.0)
        .field("body", 1.0);
    builder.add("a", 1.0, |name| match name {
        "title" => Some("keyboard".to_string()),
        "body" => Some("   ".to_string()),
        _ => None,
    });
    builder.add("b", 1.0, |name| match name {
        "title" => Some("   ".to_string()),
        "body" => Some("keyboard".to_string()),
        _ => None,
    });
    let index = builder.build();

    let results = index.search("keyboard").unwrap();
    assert_eq!(results.len(), 2);
    for result in &results {
        assert!(
            result.score.is_finite() && result.score > 0.0,
            "{} scored {}",
            result.ref_id,
            result.score
        );
    }
}

#[test]
fn corrupting_any_single_byte_never_reads_out_of_bounds() {
    // Truncation, covered in `binary_roundtrip.rs`, leaves a consistent prefix.
    // Corruption does not: flipping a byte inside a section offset points the
    // reader at data that still satisfies a length check, which is exactly where
    // unchecked slicing reads past the end.
    //
    // Stepping rather than exhausting: every byte of every index would be tens of
    // thousands of loads. The step is coprime with the 64-byte header so the walk
    // does not land on the same field alignment every time.
    let index = corruptible_index();
    let original = index.to_binary(true);

    let mut accepted = 0usize;
    let mut searched = 0usize;

    for offset in (0..original.len()).step_by(7) {
        for flip in [0x01u8, 0x80, 0xff] {
            let mut bytes = original.clone();
            bytes[offset] ^= flip;
            if let Ok(loaded) = Index::from_binary(&bytes, Arc::new(Japanese)) {
                accepted += 1;
                // A corrupted index may answer nonsense — it must not panic,
                // hang, or read out of bounds. Under a debug build the bounds
                // checks in `reader` are what make this assertion meaningful.
                if loaded.search("検索エンジン").is_ok() {
                    searched += 1;
                }
                let _ = loaded.search("機械");
                let _ = loaded.document_count();
            }
        }
    }

    // Without this the test passes trivially if the reader rejects everything —
    // which is a legal implementation, and one that would exercise none of the
    // parsing paths the test exists to cover.
    assert!(
        accepted > 20,
        "only {accepted} corrupted indexes loaded at all; the sweep is not \
         reaching the parsing paths it is meant to test"
    );
    assert!(
        searched > 20,
        "only {searched} corrupted indexes answered a query"
    );
}

#[test]
fn corrupting_the_header_is_detected_rather_than_trusted() {
    // The header carries the section offsets and counts, so it is the highest
    // leverage place for a corrupt byte. Every one of these must be rejected or
    // survive a search; silently answering from a misparsed header would mean
    // returning results computed from arbitrary bytes.
    let index = corruptible_index();
    let original = index.to_binary(true);

    let mut rejected = 0usize;
    for offset in 0..64.min(original.len()) {
        let mut bytes = original.clone();
        bytes[offset] ^= 0xff;
        match Index::from_binary(&bytes, Arc::new(Japanese)) {
            Ok(loaded) => {
                let _ = loaded.search("検索");
            }
            Err(_) => rejected += 1,
        }
    }

    // Most of the header is magic, version, counts and offsets, so a flipped
    // byte should usually be caught outright. A reader that accepted every
    // mangled header would be trusting bytes it cannot validate.
    assert!(
        rejected > 32,
        "only {rejected} of 64 header corruptions were rejected"
    );
}

#[test]
fn random_bytes_are_never_mistaken_for_an_index() {
    // A deterministic LCG rather than a dependency: the point is a broad sweep
    // of non-index bytes, and reproducibility matters more than distribution
    // quality when a failure has to be debugged.
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u8
    };

    for length in [0usize, 1, 7, 15, 16, 63, 64, 65, 128, 1024] {
        for _ in 0..64 {
            let bytes: Vec<u8> = (0..length).map(|_| next()).collect();
            // Overwhelmingly these fail the magic check. The ones that do not
            // must still not panic.
            if let Ok(loaded) = Index::from_binary(&bytes, Arc::new(English)) {
                let _ = loaded.search("keyboard");
            }
        }
    }
}

#[test]
fn a_valid_header_with_a_garbage_body_fails_cleanly() {
    // The nastiest shape: the magic and version pass, so the reader commits to
    // parsing, and everything after is wrong. This is what a partial write or a
    // truncated-then-padded transfer produces.
    let index = english_index();
    let original = index.to_binary(true);
    assert!(original.len() > 64, "index must be larger than its header");

    let mut bytes = original.clone();
    for byte in bytes[64..].iter_mut() {
        *byte = 0xff;
    }
    if let Ok(loaded) = Index::from_binary(&bytes, Arc::new(English)) {
        let _ = loaded.search("keyboard");
    }

    // And the same with zeros, which look like valid varints and empty strings
    // rather than invalid ones.
    let mut bytes = original;
    for byte in bytes[64..].iter_mut() {
        *byte = 0x00;
    }
    if let Ok(loaded) = Index::from_binary(&bytes, Arc::new(English)) {
        let _ = loaded.search("keyboard");
    }
}
