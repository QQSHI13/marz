use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use marz_core::languages::English;
use marz_core::{Index, IndexBuilder, Language};
use std::sync::Arc;

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
];

fn generate_docs(n: usize) -> Vec<(String, String, String)> {
    (0..n)
        .map(|i| {
            let title = format!("Document {} about {}", i, WORDS[i % WORDS.len()]);
            let body = format!(
                "This is the body of document {}. It contains words like {}, {}, and {}. \
                 The document explains how to use the search engine offline.",
                i,
                WORDS[(i + 1) % WORDS.len()],
                WORDS[(i + 3) % WORDS.len()],
                WORDS[(i + 5) % WORDS.len()],
            );
            (format!("doc-{}", i), title, body)
        })
        .collect()
}

fn build_index(docs: &[(String, String, String)]) -> Index {
    let language: Arc<dyn Language> = Arc::new(English);
    let mut builder = IndexBuilder::new(language);
    builder
        .ref_field("id")
        .field("title", 10.0)
        .field("body", 1.0);
    for (id, title, body) in docs {
        builder.add(id, 1.0, |name| match name {
            "title" => Some(title.clone()),
            "body" => Some(body.clone()),
            _ => None,
        });
    }
    builder.build()
}

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("build");
    for size in [100, 1_000, 5_000].iter().copied() {
        let docs = generate_docs(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &docs, |b, docs| {
            b.iter(|| build_index(black_box(docs)));
        });
    }
    group.finish();
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");
    for size in [100, 1_000, 5_000].iter().copied() {
        let docs = generate_docs(size);
        let index = build_index(&docs);
        group.bench_with_input(BenchmarkId::new("term", size), &index, |b, index| {
            b.iter(|| index.search(black_box("rust offline")).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("wildcard", size), &index, |b, index| {
            b.iter(|| index.search(black_box("doc*")).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("fuzzy", size), &index, |b, index| {
            b.iter(|| index.search(black_box("engin~1")).unwrap());
        });
    }
    group.finish();
}

fn bench_load(c: &mut Criterion) {
    let docs = generate_docs(1_000);
    let index = build_index(&docs);
    let json = index.to_json();
    let language: Arc<dyn Language> = Arc::new(English);
    c.bench_function("load_1000", |b| {
        b.iter(|| Index::load(black_box(&json), language.clone()).unwrap());
    });
}

criterion_group!(benches, bench_build, bench_search, bench_load);
criterion_main!(benches);
