#!/usr/bin/env node
// Generate the stemmer parity fixture from lunr.js.
//
//   npm install lunr@2.3.9
//   node tests/generate_stemmer_fixture.js <wordlist.txt> > tests/fixtures/stemmer_lunr_js.json
//
// lunr.js is the parity oracle for Marz's Porter stemmer. lunr.py implements
// Porter's positional rule for terminal `y` while lunr.js uses a regex that
// requires a literal non-vowel before it, so the two produce different stems
// (deploy, fly, try, ...) and cannot both be matched. The browser consumes a
// Marz index, so lunr.js wins.

const fs = require("fs");
const lunr = require("lunr");

const listPath = process.argv[2];
if (!listPath) {
  console.error("usage: generate_stemmer_fixture.js <wordlist.txt>");
  process.exit(1);
}

const words = fs
  .readFileSync(listPath, "utf8")
  .split("\n")
  .map((w) => w.trim().toLowerCase())
  .filter((w) => w.length > 0 && /^[a-z]+$/.test(w));

const out = {};
for (const w of words) {
  out[w] = lunr.stemmer(new lunr.Token(w, {})).toString();
}

const sorted = {};
for (const k of Object.keys(out).sort()) sorted[k] = out[k];
process.stdout.write(JSON.stringify(sorted, null, 0));
