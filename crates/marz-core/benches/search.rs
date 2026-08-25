//! What Marz costs, measured on the operations a caller actually performs.
//!
//! Four groups, chosen because each answers a question someone has to make a
//! decision about:
//!
//! - `build` — the docsforge build step. Scales with corpus size, and CJK is
//!   measured separately because bigram tokenization emits roughly one term per
//!   character where English emits one per word.
//! - `search` — what a keystroke costs in the browser. Split by query shape,
//!   because a wildcard scans the term dictionary while a plain term does one
//!   lookup, and CJK adds phrase verification over positions on top.
//! - `load` — what a page pays before the first search can run. This is the
//!   number the binary format exists to reduce, so it is measured against JSON
//!   rather than alone: a format that loads fast but was never compared to what
//!   it replaced proves nothing.
//! - `serialize` — the build step's other half, and the only place the
//!   positions/no-positions trade is visible as time rather than bytes.
//!
//! Run one group with `cargo bench -- search`, and compare against a previous
//! commit with `--save-baseline`/`--baseline`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use marz_core::languages::{English, Japanese};
use marz_core::{Index, IndexBuilder, Language};
use std::sync::Arc;

/// Vocabulary for the synthetic English corpus.
///
/// Deliberately more words than documents-per-shape, so term frequencies vary
/// across the corpus instead of every document containing the same terms — an
/// inverted index over a corpus where every posting list has the same length
/// would make the scorer's work look uniform when real prose is anything but.
const WORDS: &[&str] = &[
    "rust",
    "search",
    "engine",
    "offline",
    "documentation",
    "guide",
    "tutorial",
    "install",
    "configure",
    "deploy",
    "markdown",
    "python",
    "javascript",
    "wasm",
    "token",
    "index",
    "query",
    "result",
    "score",
    "rank",
    "bigram",
    "posting",
    "stemmer",
    "pipeline",
    "corpus",
    "browser",
    "dictionary",
    "segment",
    "highlight",
    "boost",
];

/// Fragments of Japanese prose, combined the same way `WORDS` is.
///
/// Real sentences rather than random characters: bigram tokenization means the
/// term distribution depends on which character *pairs* recur, and shuffled
/// characters produce a vocabulary of unique bigrams that no real corpus has.
const JA_PHRASES: &[&str] = &[
    "検索エンジンの仕組み",
    "全文検索は索引を使います",
    "形態素解析を使わない方法",
    "機械学習と人工知能",
    "統計的な学習理論の基礎",
    "日本語の文字列を正規化する",
    "ブラウザで動く軽量な検索",
    "辞書を必要としない設計",
    "転置索引の構築と圧縮",
    "文字の二文字組を単位にする",
    "位置情報を使った語句照合",
    "半角と全角の違いを吸収する",
];

struct Doc {
    id: String,
    title: String,
    body: String,
}

fn english_docs(n: usize) -> Vec<Doc> {
    (0..n)
        .map(|i| Doc {
            id: format!("doc-{i}"),
            title: format!("Document {} about {}", i, WORDS[i % WORDS.len()]),
            body: format!(
                "This is the body of document {}. It contains words like {}, {}, and {}. \
                 The document explains how to use the search engine offline, and mentions \
                 {} in passing.",
                i,
                WORDS[(i + 1) % WORDS.len()],
                WORDS[(i + 3) % WORDS.len()],
                WORDS[(i + 5) % WORDS.len()],
                WORDS[(i + 11) % WORDS.len()],
            ),
        })
        .collect()
}

fn japanese_docs(n: usize) -> Vec<Doc> {
    (0..n)
        .map(|i| Doc {
            id: format!("doc-{i}"),
            title: JA_PHRASES[i % JA_PHRASES.len()].to_string(),
            body: format!(
                "{}。{}。{}について説明します。{}。",
                JA_PHRASES[(i + 1) % JA_PHRASES.len()],
                JA_PHRASES[(i + 4) % JA_PHRASES.len()],
                JA_PHRASES[(i + 7) % JA_PHRASES.len()],
                JA_PHRASES[(i + 9) % JA_PHRASES.len()],
            ),
        })
        .collect()
}

fn build_index(docs: &[Doc], language: Arc<dyn Language>) -> Index {
    let mut builder = IndexBuilder::new(language);
    builder
        .ref_field("id")
        .field("title", 10.0)
        .field("body", 1.0);
    for doc in docs {
        builder.add(&doc.id, 1.0, |name| match name {
            "title" => Some(doc.title.clone()),
            "body" => Some(doc.body.clone()),
            _ => None,
        });
    }
    builder.build()
}

/// Corpus sizes. 5,000 is about a large documentation site; 100 is small enough
/// that per-call overhead is still visible, which is what matters for search.
const SIZES: [usize; 3] = [100, 1_000, 5_000];

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("build");
    for size in SIZES {
        // Throughput in documents, so the numbers are comparable across sizes
        // and a superlinear term shows up as falling throughput rather than
        // having to be divided out by hand.
        group.throughput(Throughput::Elements(size as u64));

        let en = english_docs(size);
        group.bench_with_input(BenchmarkId::new("en", size), &en, |b, docs| {
            b.iter(|| build_index(black_box(docs), Arc::new(English)));
        });

        // The comparison that matters: bigram tokenization emits roughly one
        // term per character, so this is expected to be slower per document.
        // How much slower is the thing worth watching.
        let ja = japanese_docs(size);
        group.bench_with_input(BenchmarkId::new("ja", size), &ja, |b, docs| {
            b.iter(|| build_index(black_box(docs), Arc::new(Japanese)));
        });
    }
    group.finish();
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");

    for size in SIZES {
        let index = build_index(&english_docs(size), Arc::new(English));

        // A plain term: one dictionary lookup, then scoring. The floor.
        //
        // Note what this query matches: "offline" is in every body, so it scores
        // the *whole corpus*. Growth here is therefore expected to be linear in
        // `size` — it is the pair with `en/rare` below that separates "cost per
        // matched document" from "cost per document in the index", and only the
        // first of those should move.
        group.bench_with_input(BenchmarkId::new("en/term", size), &index, |b, index| {
            b.iter(|| index.search(black_box("rust offline")).unwrap());
        });
        // The same shape against a term in roughly 1/30th of the corpus. Read
        // beside `en/term`: if this one grows faster than the number of matches,
        // the scorer is doing per-corpus work in a per-document loop. That is
        // exactly how the quadratic IDF bug looked before it was found — see
        // `Contribution::idf` in src/index.rs.
        group.bench_with_input(BenchmarkId::new("en/rare", size), &index, |b, index| {
            b.iter(|| index.search(black_box("bigram")).unwrap());
        });
        // A wildcard scans every term in the dictionary, so this grows with
        // vocabulary rather than with the number of matches.
        group.bench_with_input(BenchmarkId::new("en/wildcard", size), &index, |b, index| {
            b.iter(|| index.search(black_box("doc*")).unwrap());
        });
        // Fuzzy computes an edit distance per candidate term.
        group.bench_with_input(BenchmarkId::new("en/fuzzy", size), &index, |b, index| {
            b.iter(|| index.search(black_box("engin~1")).unwrap());
        });
        // Field-scoped and boosted, since those take different paths through
        // the scorer than a bare term.
        group.bench_with_input(BenchmarkId::new("en/scoped", size), &index, |b, index| {
            b.iter(|| {
                index
                    .search(black_box("title:search body:engine^5"))
                    .unwrap()
            });
        });
    }

    for size in SIZES {
        let index = build_index(&japanese_docs(size), Arc::new(Japanese));

        // Two characters is one bigram: a single posting list, no phrase
        // verification. The CJK floor.
        group.bench_with_input(BenchmarkId::new("ja/bigram", size), &index, |b, index| {
            b.iter(|| index.search(black_box("検索")).unwrap());
        });
        // Five characters is four overlapping bigrams whose posting lists must
        // be intersected and then verified against positions for adjacency.
        // This is the query shape that pays for having no dictionary.
        group.bench_with_input(BenchmarkId::new("ja/phrase", size), &index, |b, index| {
            b.iter(|| index.search(black_box("形態素解析")).unwrap());
        });
        // A query whose bigrams all occur but whose literal string never does:
        // 検索 and 索引 are both common in this corpus, 検索引 is not. So every
        // posting list is long, verification runs over all of it, and rejects.
        // That is the worst case for verification, and it is the case a user
        // typing a plausible-looking compound actually hits.
        group.bench_with_input(
            BenchmarkId::new("ja/no-phrase", size),
            &index,
            |b, index| {
                b.iter(|| index.search(black_box("検索引")).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("load");

    for size in SIZES {
        let index = build_index(&english_docs(size), Arc::new(English));
        group.throughput(Throughput::Elements(size as u64));

        // The reason the binary format exists. Measured beside the JSON it
        // replaced, because "loads in N ms" is only meaningful next to what it
        // used to cost — and JSON parsing is what a lunr-style index pays on
        // every page load.
        let json = index.to_json();
        group.bench_with_input(BenchmarkId::new("json", size), &json, |b, json| {
            b.iter(|| Index::load(black_box(json), Arc::new(English)).unwrap());
        });

        // `from_binary` materializes the postings into the same structures
        // `load` builds, so this measures parsing without JSON's cost — the
        // convenience path, and still O(index size).
        let binary = index.to_binary(true);
        group.bench_with_input(
            BenchmarkId::new("binary/materialize", size),
            &binary,
            |b, bytes| {
                b.iter(|| Index::from_binary(black_box(bytes), Arc::new(English)).unwrap());
            },
        );

        // Without positions. Fewer bytes to walk, though the saving is smaller
        // than the byte count suggests: the position data is contiguous, so
        // skipping it avoids reads rather than allocations.
        let lean = index.to_binary(false);
        group.bench_with_input(
            BenchmarkId::new("binary/materialize-no-positions", size),
            &lean,
            |b, bytes| {
                b.iter(|| Index::from_binary(black_box(bytes), Arc::new(English)).unwrap());
            },
        );

        // The number the format was designed for, and the one a browser sees:
        // `BinaryIndex::open` validates the header and computes section bounds,
        // then reads postings out of the buffer on demand. Nothing is decoded
        // up front, so this should be flat in corpus size while everything above
        // grows linearly. If it ever stops being flat, the format has acquired
        // an eager pass somewhere.
        group.bench_with_input(
            BenchmarkId::new("binary/open", size),
            &binary,
            |b, bytes| {
                b.iter(|| marz_core::BinaryIndex::open(black_box(bytes)).unwrap());
            },
        );
    }

    // A term lookup straight out of the buffer, which is what `open` defers to.
    // Paired with `open` above, these two are the whole cost of answering from a
    // mapped index without materializing it.
    let index = build_index(&english_docs(5_000), Arc::new(English));
    let bytes = index.to_binary(true);
    group.throughput(Throughput::Elements(1));
    group.bench_function("binary/term-lookup", |b| {
        let binary = marz_core::BinaryIndex::open(&bytes).unwrap();
        b.iter(|| {
            let id = binary.term_id(black_box("search")).unwrap().unwrap();
            binary.postings(id).unwrap()
        });
    });

    // CJK indexes are the ones large enough for load time to be noticeable, so
    // measure the format on the corpus it was sized for.
    let ja = build_index(&japanese_docs(5_000), Arc::new(Japanese));
    let ja_bytes = ja.to_binary(true);
    group.throughput(Throughput::Elements(5_000));
    group.bench_function("binary/materialize-ja-5000", |b| {
        b.iter(|| Index::from_binary(black_box(&ja_bytes), Arc::new(Japanese)).unwrap());
    });
    group.bench_function("binary/open-ja-5000", |b| {
        b.iter(|| marz_core::BinaryIndex::open(black_box(&ja_bytes)).unwrap());
    });

    group.finish();
}

fn bench_serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialize");

    let index = build_index(&english_docs(5_000), Arc::new(English));
    group.bench_function("json", |b| b.iter(|| black_box(&index).to_json()));
    group.bench_function("binary", |b| b.iter(|| black_box(&index).to_binary(true)));
    group.bench_function("binary/no-positions", |b| {
        b.iter(|| black_box(&index).to_binary(false))
    });

    let ja = build_index(&japanese_docs(5_000), Arc::new(Japanese));
    group.bench_function("binary/ja", |b| b.iter(|| black_box(&ja).to_binary(true)));

    group.finish();
}

criterion_group!(
    benches,
    bench_build,
    bench_search,
    bench_load,
    bench_serialize
);
criterion_main!(benches);
