use std::sync::Arc;

use marz_core::language::MultiLanguage;
use marz_core::languages::{Chinese, English, Japanese, Korean};
use marz_core::{IndexBuilder, Language};

fn build_chinese_index() -> marz_core::Index {
    let language: Arc<dyn Language> = Arc::new(Chinese);
    let mut builder = IndexBuilder::new(language);
    builder
        .ref_field("id")
        .field("title", 10.0)
        .field("body", 1.0);
    builder.add("doc-zh-1", 1.0, |name| match name {
        "title" => Some("中文文档".to_string()),
        "body" => Some("这是一个中文文档，用于测试中文搜索。".to_string()),
        _ => None,
    });
    builder.add("doc-zh-2", 1.0, |name| match name {
        "title" => Some("Rust 编程语言".to_string()),
        "body" => Some("Rust 是一门快速的系统编程语言。".to_string()),
        _ => None,
    });
    builder.build()
}

fn build_japanese_index() -> marz_core::Index {
    let language: Arc<dyn Language> = Arc::new(Japanese);
    let mut builder = IndexBuilder::new(language);
    builder.ref_field("id").field("title", 1.0);
    builder.add("doc-ja-1", 1.0, |name| match name {
        "title" => Some("日本語の検索".to_string()),
        _ => None,
    });
    builder.build()
}

fn build_korean_index() -> marz_core::Index {
    let language: Arc<dyn Language> = Arc::new(Korean);
    let mut builder = IndexBuilder::new(language);
    builder.ref_field("id").field("title", 1.0);
    builder.add("doc-ko-1", 1.0, |name| match name {
        "title" => Some("한국어 검색".to_string()),
        _ => None,
    });
    builder.build()
}

fn build_mixed_index() -> marz_core::Index {
    let language = Arc::new(MultiLanguage::new(vec![
        Arc::new(Chinese) as Arc<dyn Language>,
        Arc::new(English),
    ]));
    let mut builder = IndexBuilder::new(language);
    builder
        .ref_field("id")
        .field("title", 1.0)
        .field("body", 1.0);
    builder.add("doc-mixed", 1.0, |name| match name {
        "title" => Some("Rust 中文文档".to_string()),
        "body" => Some("This is a Rust document in Chinese.".to_string()),
        _ => None,
    });
    builder.build()
}

#[test]
fn manual_chinese_search() {
    let index = build_chinese_index();

    for query in ["中文", "文档", "搜索", "rust"] {
        let results = index.search(query).unwrap();
        println!(
            "Chinese query '{}': {:?}",
            query,
            results.iter().map(|r| &r.ref_id).collect::<Vec<_>>()
        );
        assert!(!results.is_empty(), "expected results for '{}'", query);
    }

    // Title-boosted doc should rank higher for a term that appears in both title and body.
    let results = index.search("中文").unwrap();
    assert_eq!(results[0].ref_id, "doc-zh-1");
}

#[test]
fn manual_japanese_search() {
    let index = build_japanese_index();
    let results = index.search("日本語").unwrap();
    println!(
        "Japanese query '日本語': {:?}",
        results.iter().map(|r| &r.ref_id).collect::<Vec<_>>()
    );
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].ref_id, "doc-ja-1");
}

#[test]
fn manual_korean_search() {
    let index = build_korean_index();
    let results = index.search("한국어").unwrap();
    println!(
        "Korean query '한국어': {:?}",
        results.iter().map(|r| &r.ref_id).collect::<Vec<_>>()
    );
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].ref_id, "doc-ko-1");
}

#[test]
fn manual_mixed_language_search() {
    let index = build_mixed_index();

    let rust_results = index.search("rust").unwrap();
    println!(
        "Mixed query 'rust': {:?}",
        rust_results.iter().map(|r| &r.ref_id).collect::<Vec<_>>()
    );
    assert_eq!(rust_results.len(), 1);

    let zh_results = index.search("中文").unwrap();
    println!(
        "Mixed query '中文': {:?}",
        zh_results.iter().map(|r| &r.ref_id).collect::<Vec<_>>()
    );
    assert_eq!(zh_results.len(), 1);
}

#[test]
fn manual_wildcards_and_fuzzy() {
    let language: Arc<dyn Language> = Arc::new(English);
    let mut builder = IndexBuilder::new(language);
    builder.ref_field("id").field("title", 1.0);
    builder.add("a", 1.0, |_| Some("foobar".to_string()));
    builder.add("b", 1.0, |_| Some("hello world".to_string()));
    let index = builder.build();

    let trailing = index.search("foo*").unwrap();
    println!(
        "Wildcard 'foo*': {:?}",
        trailing.iter().map(|r| &r.ref_id).collect::<Vec<_>>()
    );
    assert_eq!(trailing.len(), 1);
    assert_eq!(trailing[0].ref_id, "a");

    let leading = index.search("*bar").unwrap();
    println!(
        "Wildcard '*bar': {:?}",
        leading.iter().map(|r| &r.ref_id).collect::<Vec<_>>()
    );
    assert_eq!(leading.len(), 1);
    assert_eq!(leading[0].ref_id, "a");

    let fuzzy = index.search("helo~1").unwrap();
    println!(
        "Fuzzy 'helo~1': {:?}",
        fuzzy.iter().map(|r| &r.ref_id).collect::<Vec<_>>()
    );
    assert_eq!(fuzzy.len(), 1);
    assert_eq!(fuzzy[0].ref_id, "b");
}

#[test]
fn manual_presence_and_boost() {
    let language: Arc<dyn Language> = Arc::new(English);
    let mut builder = IndexBuilder::new(language);
    builder
        .ref_field("id")
        .field("title", 10.0)
        .field("body", 1.0);
    builder.add("a", 1.0, |name| match name {
        "title" => Some("marz search".to_string()),
        "body" => Some("fast offline engine".to_string()),
        _ => None,
    });
    builder.add("b", 1.0, |name| match name {
        "title" => Some("offline engine".to_string()),
        "body" => Some("marz is fast".to_string()),
        _ => None,
    });
    let index = builder.build();

    let required = index.search("+marz +engine").unwrap();
    println!(
        "Required '+marz +engine': {:?}",
        required.iter().map(|r| &r.ref_id).collect::<Vec<_>>()
    );
    assert_eq!(required.len(), 2);

    let prohibited = index.search("marz -offline").unwrap();
    println!(
        "Prohibited 'marz -offline': {:?}",
        prohibited.iter().map(|r| &r.ref_id).collect::<Vec<_>>()
    );
    assert_eq!(prohibited.len(), 0);

    let boosted = index.search("title:marz^5 body:marz").unwrap();
    println!(
        "Boosted title:marz^5 body:marz scores: {:?}",
        boosted
            .iter()
            .map(|r| (r.ref_id.clone(), r.score))
            .collect::<Vec<_>>()
    );
    assert_eq!(boosted[0].ref_id, "a");
}

#[test]
fn manual_empty_and_stop_word_queries() {
    let language: Arc<dyn Language> = Arc::new(English);
    let mut builder = IndexBuilder::new(language);
    builder.ref_field("id").field("text", 1.0);
    builder.add("a", 1.0, |_| Some("marz search".to_string()));
    let index = builder.build();

    let empty = index.search("").unwrap();
    println!("Empty query results: {}", empty.len());
    assert!(empty.is_empty());

    let stop = index.search("the").unwrap();
    println!("Stop-word query 'the' results: {}", stop.len());
    assert!(stop.is_empty());
}
