# Marz

A search engine for documentation sites that need CJK to work, and can't ship a
dictionary to get it.

Build the index once — in Python, at build time — and search it in the browser
from WebAssembly. No server, no network calls at query time, no segmentation
model to download.

```python
import marz

builder = marz.IndexBuilder("ja", ref_field="location")
builder.field("title", 10.0)
builder.field("text")
builder.add({"location": "guide/", "title": "検索エンジン", "text": "..."})

Path("search.bin").write_bytes(builder.build().to_bytes())
```

```javascript
import { initialize, load } from "marz-search";

await initialize();
const index = await load(await fetch("/search.bin").then((r) => r.arrayBuffer()));
for (const hit of index.search("検索エンジン")) {
  console.log(hit.ref, hit.score, hit.matches);
}
```

## Why

The usual way to search Chinese or Japanese text is a segmentation dictionary —
jieba is 4.9 MB uncompressed, and TinySegmenter is smaller but still a model
that has to be right about Japanese. Either way a browser downloads it before
the first search.

Marz tokenizes CJK into **overlapping character bigrams** instead. `検索エンジン`
becomes `検索`, `索エ`, `エン`, `ンジ`, `ジン` — on both sides, so a query
matches whatever a document contains without either side knowing where words
begin. Nothing to download, nothing to be wrong about.

Bigrams alone over-match: a document containing 検索 and エンジン separately
matches a query for 検索エンジン. So Marz records term positions and verifies
adjacency at query time, which **boosts** a real phrase match rather than
filtering the partial one. A document with the phrase scores roughly two orders
of magnitude above one with the pieces scattered — it sorts to the top, and the
partial hit stays available instead of vanishing. Filtering would make a CJK
query behave as AND while the same query in English behaves as OR, which is a
worse surprise than a low-ranked extra result.

Korean is written with spaces, so it is tokenized on whitespace. Forcing bigrams
on a language that does not need them would only cost index size.

## Numbers

Measured on this machine, on a synthetic corpus of 5,000 documents with a title
and a body. Reproduce with `just size` and `cargo bench`.

| | English | Japanese |
|---|---|---|
| JSON index (what lunr ships) | 3.5 MB | 7.4 MB |
| Binary index | 716 KB (20%) | 1.2 MB (17%) |
| Binary, positions dropped | 493 KB (14%) | 782 KB (11%) |

The binary format is read in place. `BinaryIndex::open` validates the header and
computes section bounds in **~330 ns regardless of corpus size** — postings are
decoded from the buffer as queries ask for them. Parsing the equivalent JSON
takes ~112 ms at 5,000 documents, and materializing the whole binary index into
owned structures takes ~52 ms. Dropping positions costs highlighting and phrase
verification.

Search cost tracks how many documents match, not how many exist. A selective
query over the 5,000-document English index takes **~3 ms**; a query for a term
that appears in every document takes ~24 ms, because scoring and sorting the
entire corpus is the actual work being asked for. Japanese is ~24 ms for a
single bigram and ~40 ms for a five-character phrase, where the extra time is
adjacency verification over positions. Reproduce with `cargo bench -- search`.

WebAssembly, after `wasm-opt`: **175 KB raw, 84 KB gzipped**. That is the whole
search engine including all four languages. The optional client-side index
builder adds 39 KB and is behind a feature flag, because a site that builds its
index in Python does not need it in the browser.

## Dependencies

`marz-core` depends on `serde` and `serde_json`, and only for the legacy JSON
index format — the binary format, the tokenizers, the Porter stemmer and the
query parser are all hand-written with no dependencies. The stemmer is not
regex-based, which is what keeps it out of the WASM bundle.

## Languages

| Code | Tokenization | Stemming |
|---|---|---|
| `en` | whitespace + punctuation | Porter, matching lunr.js |
| `zh` | Han bigrams | none |
| `ja` | Han/Kana bigrams | none |
| `ko` | whitespace | none |

A single index can serve several languages at once via `MultiLanguage`, which
dispatches per script — one index for a site with translated pages, rather than
one per locale. Text is NFC-normalized and width-folded first, so fullwidth
`ＲＵＳＴ` is found by typing `rust` and halfwidth `ｶﾞｲﾄﾞ` by typing `ガイド`.

## Relationship to lunr

Marz began as a lunr reimplementation and no longer is one. lunr scores through
a precomputed field-vector cosine similarity; Marz sums BM25 weights at query
time, which is what makes the index small enough to read in place. The query
language — `+required`, `-prohibited`, `field:scoped`, `term^boost`, `fuzzy~1`,
`wild*` — is lunr's, and English **ranking** is checked against lunr.js on every
commit.

Absolute scores deliberately differ. lunr divides by the query vector's
magnitude, which is constant across documents for a given query and so cannot
change their order; ranking is the observable behaviour, and pinning the
magnitude would freeze a design decision this rewrite reversed.

The oracle is lunr.**js**, not lunr.py. The two stem terminal `y` differently —
`fly` becomes `fli` under lunr.js and stays `fly` under lunr.py — so a query for
`fly` finds a document containing only `flies` under one and nothing under the
other. Both cannot be matched. Marz follows lunr.js, because a browser consuming
a Marz index sits beside lunr.js. See `crates/marz-core/tests/golden.rs`.

## Layout

- `crates/marz-core` — the engine: tokenizers, stemmer, query parser, BM25
  scorer, binary format
- `crates/marz-wasm` — wasm-bindgen bindings, search-only by default
- `js/` — TypeScript wrapper around the WASM module
- `python/` — PyO3 bindings for the build step
- `tests/fixtures` — the lunr.js ranking oracle, and the hand-written CJK
  expectations that have no oracle

## Development

```
just test-all     # Rust, Python and JS suites
just lint         # clippy, including the wasm-only builder feature
just size         # shipped WASM size, raw and gzipped
just golden       # regenerate the lunr.js ranking oracle
cargo bench       # build, search, load and serialize
```

## License

Apache-2.0
