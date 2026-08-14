#!/usr/bin/env python3
"""Generate golden fixtures from lunr.py for Marz parity tests."""

from __future__ import annotations

import json
from pathlib import Path

import lunr

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"


def save(name: str, data: object) -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {path}")


def main() -> None:
    documents = [
        {
            "location": "page-a/",
            "title": "Installing Marz",
            "text": "Marz is a fast offline search engine. Install it with pip.",
        },
        {
            "location": "page-b/",
            "title": "Search query syntax",
            "text": "Use wildcards like foo* and fuzzy matches like hello~1.",
        },
        {
            "location": "page-c/",
            "title": "中文文档",
            "text": "这是一个中文文档，用于测试中文搜索。",
        },
        {
            "location": "page-d/",
            "title": "Running fast",
            "text": "Running is faster than walking. Walk before you run.",
        },
        {
            "location": "page-e/",
            "title": "The quick brown fox",
            "text": "The quick brown fox jumps over the lazy dog.",
        },
        {
            "location": "page-f/",
            "title": "Boosted foobar",
            "text": "This document has a foobar term and is boosted.",
        },
    ]

    documents_with_boosts = []
    for doc in documents:
        boosted_doc = dict(doc)
        if doc["location"] == "page-f/":
            boosted_doc["_boost"] = 2
        documents_with_boosts.append(boosted_doc)
    save("documents.json", documents_with_boosts)

    builder = lunr.get_default_builder()
    builder.ref("location")
    builder.field("title", boost=10)
    builder.field("text")

    for doc in documents:
        if doc["location"] == "page-f/":
            builder.add(doc, {"boost": 2})
        else:
            builder.add(doc)

    index = builder.build()
    serialized = index.serialize()
    save("index.json", serialized)

    queries = [
        # basic term matching and stemming
        "marz",
        "install",
        "installing",
        "running",
        "walk",
        # stop word only
        "the",
        # wildcards
        "foo*",
        "*bar",
        # fuzzy
        "hello~1",
        "helo~1",
        # field scoping
        "title:search",
        "text:foobar",
        # required / prohibited presence
        "+marz +engine",
        "+search +wildcards",
        "marz -offline",
        "-marz",
        # boost
        "foobar^5",
        # multi-term OR ranking
        "marz search",
    ]
    results: dict[str, list[dict]] = {}
    for query in queries:
        results[query] = [
            {"ref": r["ref"], "score": round(r["score"], 6)} for r in index.search(query)
        ]

    save("queries.json", results)


if __name__ == "__main__":
    main()
