# What lunr does, and what Marz decided to do instead

Research notes from reading `lunr.js`, `lunr.py`, `lunr-languages` and `jieba`,
cloned on 2026-08-14. Kept because it records *why* Marz is built the way it is —
the alternatives here were investigated and rejected, and without the notes a
future reader can only see the decision, not the reasoning.

Historical. This is not a specification of Marz, and where the two disagree, the
code is right. Marz began as a lunr reimplementation and the rewrite reversed
several of the choices below on purpose.

## Lunr's scoring model

Lunr does **not** use pure TF-IDF. It uses BM25 term weights inside a
vector-space cosine similarity with asymmetric normalization.

Build-time field vector weight for term `t` in field `f` of document `d`:

```
idf(t)   = log(1 + abs((N - df(t) + 0.5) / (df(t) + 0.5)))
w(t,d,f) = idf(t) * ((k1 + 1) * tf(t,d,f)) / (k1 * (1 - b + b * (|f_d| / avg(|f|))) + tf(t,d,f))
w(t,d,f) *= field_boost(f) * doc_boost(d)
w(t,d,f)  = round(w, 3)
```

Query-time score for a matching field:

```
field_score = dot(query_vector[field], field_vector[docRef/fieldName]) / magnitude(query_vector[field])
doc_score   = sum of field_score across matching fields
```

The denominator is the **query** vector's magnitude, not the field vector's.

**Marz does not replicate this.** The weight formula is the same BM25, but Marz
stores term frequencies and evaluates the weight at query time instead of
precomputing and rounding a field vector per document-field. Two reasons:

- A field vector per document-field is the bulk of a lunr index. Storing `tf` as
  a varint instead is what makes the binary format small enough to read in
  place, which is the whole point of the format.
- The query-magnitude division is constant across documents for a given query,
  so it cannot change their order. It changes score *magnitudes* only, and
  pinning it would freeze a design decision the rewrite reversed.

So Marz asserts **ranking** parity against lunr.js, not score equality. See
`crates/marz-core/tests/golden.rs`.

## Lunr's index structure

- `invertedIndex[term]`:
  - `_index`: dense integer term index
  - `fieldName -> { docRef -> { position: [[start, len], ...] } }`
- `fieldVectors["docRef/fieldName"]`: sparse vector `[termIndex, score, ...]`
- `tokenSet`: DFA/trie of all terms, for wildcard and fuzzy expansion
- `pipeline`: registered function labels, so a reloaded index re-runs the same
  pipeline on queries

Marz keeps the inverted index and the token set, drops the field vectors, and
adds a corpus-statistics section (field lengths, boosts, `k1`/`b`) that scoring
needs now that it happens at query time.

## Query language

The lexer emits `FIELD`, `TERM`, `EDIT_DISTANCE` (`~N`), `BOOST` (`^N`) and
`PRESENCE` (`+`/`-`); the parser builds clauses carrying `term`, `fields`,
`boost`, `editDistance`, `wildcard`, `presence` and `usePipeline`. Terms
containing `*` bypass the pipeline, and `~N` expands through a Levenshtein
automaton over the token set.

**Marz reimplements this as-is.** It is a good query language, users of
lunr-backed documentation sites already know it, and none of it constrains the
index format.

## CJK findings

What `lunr-languages` and its neighbours actually ship:

| Language | Approach | Cost |
|---|---|---|
| Chinese | `@node-rs/jieba` or `Intl.Segmenter`, falling back to Han bigrams | jieba's `dict.txt` is 4.9 MB uncompressed |
| Japanese | `TinySegmenter` (`tinyseg.js`) | ~22 KB |
| Korean | nothing beyond generic CJK handling | — |
| Thai/Hindi | `wordcut.js` | ~677 KB |

Chinese has no stemming and a stop-word list of about 100 characters.

These are the numbers that motivated the project. A browser downloads the
segmentation model before it can answer the first query, and `Intl.Segmenter`
is not a way out — it moves the dictionary into the engine rather than removing
it, and what it segments varies by platform.

## The bigram decision

Lunr uses overlapping Han bigrams only as a *fallback* when no segmenter is
available, and the conventional reading — which these notes originally
recorded — is that bigrams are adequate for Chinese and poor for Japanese.

Marz uses bigrams for both anyway, and the reason the conventional objection
does not bite is **phrase verification**. Bigrams over-match: a document
containing 検索 and エンジン in unrelated sentences matches a query for
検索エンジン. Marz records term positions and checks adjacency at query time,
which **boosts** a genuine phrase match rather than filtering the partial one.
The margin is large — `crates/marz-core/tests/cjk.rs` asserts a real phrase
outranks a scattered partial by more than 20×, and in practice it is closer to
two orders of magnitude — so the phrase hit sorts to the top and the partial
stays reachable.

Boosting rather than filtering is deliberate. Filtering would make a CJK query
behave as AND while the same query in English behaves as OR, which is a worse
surprise for a user than one extra low-ranked result.

What this costs: bigram tokenization emits roughly one term per character where
word tokenization emits one per word, so a Japanese index is larger than an
English one over a comparable corpus — 1.2 MB against 716 KB at 5,000 synthetic
documents (`just size`). No segmented Japanese index was built for comparison,
so the cost *against a segmenter* is not measured here. Positions are also
required for verification, so they cannot be dropped for CJK the way they can
for English-only indexes. That trade bought a zero-byte segmentation model,
which was the higher priority.

Korean is written with spaces, so it is tokenized on whitespace. Forcing
bigrams on a language that does not need them would only cost index size.

## What was rejected

- **Trigrams, or script-aware segmentation on top of bigrams.** More index size
  for precision that phrase verification already recovers.
- **A compressed bundled dictionary for higher Chinese quality.** Compression
  changes the download from megabytes to hundreds of kilobytes; it does not
  change the fact that there is a dictionary to be wrong, to update, and to
  download. "No dictionary" is the product claim.
- **A TinySegmenter port for Japanese.** Same reasoning: 22 KB is small, but it
  is a model that has to be right about Japanese, and phrase verification made
  it unnecessary.
- **Snowball stemmers for European languages.** Not rejected on merit — simply
  not built. English is the only stemmed language today, and its Porter
  implementation is hand-written specifically to keep a regex engine out of the
  WASM bundle.
