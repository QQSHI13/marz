# Lunr analysis for Marz

Source: cloned `lunr.js`, `lunr.py`, `lunr-languages`, and `jieba` into `~/lunr` on 2026-08-14.

## The exact scoring model

Lunr does **not** use pure TF-IDF. It uses **BM25 term weights inside a vector-space cosine similarity with asymmetric normalization**.

Build-time field vector weight for term `t` in field `f` of document `d`:

```
idf(t)  = log(1 + abs((N - df(t) + 0.5) / (df(t) + 0.5)))
w(t,d,f) = idf(t) * ((k1 + 1) * tf(t,d,f)) / (k1 * (1 - b + b * (|f_d| / avg(|f|))) + tf(t,d,f))
w(t,d,f) *= field_boost(f) * doc_boost(d)
w(t,d,f)  = round(w, 3)
```

Query-time score for a matching field:

```
field_score = dot(query_vector[field], field_vector[docRef/fieldName]) / magnitude(query_vector[field])
doc_score   = sum of field_score across matching fields
```

**Critical detail**: the denominator is the **query vector magnitude only**, not the field vector magnitude. Replicating lunr exactly requires this asymmetry.

## Index structure

- `invertedIndex[term]`:
  - `_index`: dense integer term index
  - `fieldName -> { docRef -> { position: [[start, len], ...] } }`
- `fieldVectors["docRef/fieldName"]`: sparse vector `[termIndex, score, ...]`
- `tokenSet`: DFA/trie of all terms for wildcard/fuzzy expansion
- `pipeline`: registered function labels for reload

## Query language

Lexer emits: `FIELD`, `TERM`, `EDIT_DISTANCE` (`~N`), `BOOST` (`^N`), `PRESENCE` (`+`/`-`).

Parser builds clauses with:
- `term`, `fields`, `boost`, `editDistance`, `wildcard`, `presence`, `usePipeline`

Terms with `*` bypass the pipeline. `~N` uses a Levenshtein automaton.

## CJK findings

- **Chinese**: `lunr-languages` uses `@node-rs/jieba` or `Intl.Segmenter`; falls back to overlapping Han bigrams. Stop word list ~100 chars; no stemming.
- **Japanese**: uses `TinySegmenter` (`tinyseg.js`, ~22 KB). Bigrams alone are poor for Japanese.
- **Korean**: no special handling in lunr-languages beyond generic CJK.
- **Thai/Hindi**: uses `wordcut.js` (~677 KB).
- **jieba dict.txt**: 4.9 MB uncompressed. Too large to bundle raw; needs compression or truncation if used.

## Bigrams do work — partially

Lunr itself uses Han bigrams as a Chinese fallback. They are dictionary-free and offline, but:
- Miss non-aligned multi-char words.
- Increase index size.
- Work best for Chinese, poorly for Japanese.

A better baseline: script-aware segmentation + overlapping bigrams/trigrams. Optional compressed dictionary for higher Chinese quality.

## Implementation checklist for marz

1. Replicate BM25 field-vector scoring with asymmetric query normalization.
2. Build inverted index + field vectors in Rust; serialize to compact binary.
3. Port lunr's query lexer/parser.
4. Port `TokenSet` for wildcard/fuzzy expansion.
5. Pipeline: trimmer, stopWordFilter, stemmer registry; default English Porter stemmer.
6. European languages: Snowball stemmers.
7. Chinese: script-aware bigrams/trigrams + optional compressed dictionary.
8. Japanese: TinySegmenter port or equivalent compact segmenter.
9. Korean: Hangul syllable/jamo tokenizer.
10. Expose identical behavior through Python (build) and WASM (search) bindings.
