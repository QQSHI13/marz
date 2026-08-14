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
    ]

    save("documents.json", documents)

    builder = lunr.get_default_builder()
    builder.ref("location")
    builder.field("title", boost=10)
    builder.field("text")

    for doc in documents:
        builder.add(doc)

    index = builder.build()
    serialized = index.serialize()
    save("index.json", serialized)

    queries = [
        "marz",
        "install",
        "foo*",
        "hello~1",
        "title:search",
        "+marz +engine",
        "中文",
    ]
    results: dict[str, list[dict]] = {}
    for query in queries:
        results[query] = [
            {"ref": r["ref"], "score": round(r["score"], 6)} for r in index.search(query)
        ]

    save("queries.json", results)


if __name__ == "__main__":
    main()
