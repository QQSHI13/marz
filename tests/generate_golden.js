#!/usr/bin/env node
// Generate the English ranking oracle from lunr.js.
//
//   just golden
//
// or by hand:
//
//   cd tests && npm install && node generate_golden.js
//
// lunr is pinned in tests/package.json, which exists only for this script — it
// is not part of the published js/ package and nothing at runtime depends on it.
//
// Writes tests/fixtures/documents.json, queries.json and oracle.json.
//
// lunr.js, not lunr.py, is the oracle. The two implement Porter differently for
// terminal `y` — lunr.py follows Porter's positional rule, lunr.js uses a regex
// requiring a literal non-vowel — so they stem `deploy`, `fly` and `try`
// differently and cannot both be matched. Marz follows lunr.js because a browser
// consuming a Marz index sits beside lunr.js, and the stemmer parity fixture in
// `stemmer_lunr_js.json` is generated from it too. Generating the ranking oracle
// from lunr.py while the stemmer targets lunr.js would mean the two fixtures
// disagreed about what a term even is.

const fs = require("fs");
const path = require("path");
const lunr = require("lunr");

const FIXTURES = path.join(__dirname, "fixtures");

/**
 * The corpus. Each entry is chosen to exercise something specific, and the
 * comments say what — a document nobody can explain the purpose of is a document
 * nobody can safely change.
 */
const DOCUMENTS = [
  {
    location: "page-a/",
    title: "Installing Marz",
    text: "Marz is a fast offline search engine. Install it with pip.",
    // install/installing: stemming collapses the query and the document.
  },
  {
    location: "page-b/",
    title: "Search query syntax",
    text: "Use wildcards like foo* and fuzzy matches like hello~1.",
    // Contains the operator characters as literal text, so a tokenizer that
    // treats them as syntax everywhere would lose them here.
  },
  {
    location: "page-c/",
    title: "中文文档",
    text: "这是一个中文文档，用于测试中文搜索。",
    // CJK in an English index. lunr.js splits on whitespace, so this becomes a
    // handful of long tokens; the point is that it does not crash or match
    // anything English. Real CJK behaviour is `crates/marz-core/tests/cjk.rs`.
  },
  {
    location: "page-d/",
    title: "Running fast",
    text: "Running is faster than walking. Walk before you run.",
    // run/running/ran and walk/walking in one document: term frequency after
    // stemming, not just presence.
  },
  {
    location: "page-e/",
    title: "The quick brown fox",
    text: "The quick brown fox jumps over the lazy dog.",
    // Stop words in both fields, and the only document where "the" appears in a
    // title — so a broken stop-word filter shows up as a title-boosted hit.
  },
  {
    location: "page-f/",
    title: "Boosted foobar",
    text: "This document has a foobar term and is boosted.",
    // Carries a document boost, below.
  },
  {
    location: "page-g/",
    title: "Deploying and flying",
    text: "Deploy the fleet. Flies fly, and we tried to deploy.",
    // The terminal-`y` words lunr.js and lunr.py stem differently. This is here
    // so that regenerating from the wrong oracle changes the fixture visibly
    // rather than silently agreeing.
  },
];

/** Document boosts, keyed by ref. Absent means 1. */
const BOOSTS = { "page-f/": 2 };

const QUERIES = [
  // Plain terms and stemming.
  "marz",
  "install",
  "installing",
  "running",
  "walk",
  // Terminal `y`, where the two lunr implementations diverge.
  "deploy",
  "deploying",
  "fly",
  "flies",
  "tried",
  // Stop word only: must return nothing, not everything.
  "the",
  // Wildcards, leading and trailing.
  "foo*",
  "*bar",
  "*oo*",
  // Fuzzy. The same misspelling at two distances, so a difference between them
  // isolates the distance parameter rather than the spelling: "ofline" is one
  // deletion from "offline", so ~1 reaches page-a and ~0 must not.
  //
  // Not "hello~1": the corpus contains the literal text "hello~1." as an example
  // of the syntax, which lunr indexes as the term `hello~1`. A query for
  // `hello~1` is then a fuzzy search for "hello" and can never match it, so the
  // case looks like it tests fuzzy matching while asserting nothing.
  "ofline~1",
  "ofline~0",
  "instal~1",
  // Field scoping.
  "title:search",
  "text:foobar",
  "title:foobar",
  // Presence: required, prohibited, and both.
  "+marz +engine",
  "+search +wildcards",
  "marz -offline",
  "-marz",
  // page-e is the only fox document and it also contains dog, so the second of
  // these is empty. The first is here to prove that is the prohibition working
  // rather than the requirement matching nothing.
  "+fox",
  "+fox -dog",
  // Term boost.
  "foobar^5",
  // Multi-term OR, where ranking is the whole assertion.
  "marz search",
  "run walk fast",
  // Nothing matches.
  "xylophone",
];

function main() {
  fs.mkdirSync(FIXTURES, { recursive: true });

  // documents.json carries the boost as `_boost` so the Rust side can rebuild
  // the same index. It is not a field lunr sees.
  const withBoosts = DOCUMENTS.map((doc) => {
    const out = { ...doc };
    if (BOOSTS[doc.location]) out._boost = BOOSTS[doc.location];
    return out;
  });
  write("documents.json", withBoosts, 2);

  const index = lunr(function () {
    this.ref("location");
    this.field("title", { boost: 10 });
    this.field("text");
    // lunr's default pipeline is trimmer → stopWordFilter → stemmer, which is
    // what Marz's English pipeline reproduces. Left alone deliberately.
    for (const doc of DOCUMENTS) {
      const boost = BOOSTS[doc.location];
      this.add(doc, boost ? { boost } : undefined);
    }
  });

  const results = {};
  for (const query of QUERIES) {
    results[query] = index
      .search(query)
      .map((r) => ({ ref: r.ref, score: Number(r.score.toFixed(6)) }));
  }
  write("queries.json", results, 2);

  // The oracle's identity, so a fixture regenerated from a different version or
  // a different implementation is visible in the diff rather than inferred.
  write(
    "oracle.json",
    {
      implementation: "lunr.js",
      version: lunr.version,
      note:
        "Ranking oracle. Scores are lunr's asymmetric cosine similarity and " +
        "are recorded for reference only — Marz asserts ranking, not " +
        "magnitude. See crates/marz-core/tests/golden.rs.",
    },
    2,
  );
}

function write(name, data, indent) {
  const file = path.join(FIXTURES, name);
  fs.writeFileSync(file, JSON.stringify(data, null, indent) + "\n", "utf8");
  console.error(`wrote ${file}`);
}

main();
