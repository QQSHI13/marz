"""Marz — an offline search engine with first-class CJK support.

Build an index in Python, ship it to a browser, search it with WebAssembly.

    import marz

    b = marz.IndexBuilder("ja", ref_field="location")
    b.field("title", 10.0)
    b.field("text")
    b.add({"location": "guide/intro", "title": "検索エンジン", "text": "..."})

    index = b.build()
    for hit in index.search("検索"):
        print(hit.ref, hit.score)

    Path("search.bin").write_bytes(index.to_bytes())

CJK text is tokenized into overlapping character bigrams, so there is no
segmentation dictionary to install and no model to load. `tokenize` shows what
a string actually splits into, which is the quickest way to understand a
surprising result.
"""

try:
    from ._marz import (
        FormatError,
        Index,
        IndexBuilder,
        QueryError,
        Result,
        __version__,
        index_language,
        languages,
        tokenize,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Marz native extension is not built. Run `maturin develop` in python/."
    ) from exc

__all__ = [
    "FormatError",
    "Index",
    "IndexBuilder",
    "QueryError",
    "Result",
    "__version__",
    "index_language",
    "languages",
    "tokenize",
]
