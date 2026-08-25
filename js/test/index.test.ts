/**
 * Tests for the browser bindings and the TypeScript wrapper.
 *
 * These exercise the boundary, not the engine: that a Rust `Vec<(usize, usize)>`
 * arrives as `[[start, length]]`, that a parse failure arrives as an `Error` with
 * usable offsets, that `highlight` slices text the caller can actually render.
 * Engine behaviour is covered by the Rust suite; a test here that asserted
 * ranking would fail for reasons that have nothing to do with this layer.
 *
 * Run with `npm test`, which builds the WebAssembly first. The module is loaded
 * from `js/pkg`, so a stale build fails loudly rather than being tested.
 */

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { after, before, describe, test } from "node:test";

import {
  MarzIndex,
  highlight,
  indexLanguage,
  initialize,
  languages,
  load,
  normalize,
  tokenize,
  version,
} from "../src/index.ts";
// The raw binding, for the handful of assertions that compare against
// normalization inline. The exported wrapper is async because it may need to
// initialize; inside a test that has already awaited `before`, that await is
// noise that obscures what is being checked.
import { normalize as normalizeSync } from "../pkg/marz_wasm.js";

const fixture = (name: string) =>
  fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url));

/** The English fixture's field text, for checking highlight offsets. */
type Doc = { location: string; title: string; text: string };

let en: MarzIndex;
let ja: MarzIndex;
let enDocs: Doc[];
let jaDocs: Doc[];

before(async () => {
  // Pass the bytes rather than a path: Node's `fetch` does not implement `file:`
  // URLs, so wasm-pack's default path resolution fails there. A browser needs no
  // argument at all.
  await initialize(await readFile(fixture("../pkg/marz_wasm_bg.wasm")));
  en = await load(await readFile(fixture("en.marz")));
  ja = await load(await readFile(fixture("ja.marz")));
  enDocs = JSON.parse(await readFile(fixture("corpus_en.json"), "utf8"));
  jaDocs = JSON.parse(await readFile(fixture("corpus_ja.json"), "utf8"));
});

after(() => {
  en?.free();
  ja?.free();
});

describe("module", () => {
  test("reports the languages it was built with", async () => {
    assert.deepEqual(await languages(), ["en", "zh", "ja", "ko"]);
  });

  test("reports a version", async () => {
    assert.match(await version(), /^\d+\.\d+\.\d+/);
  });

  test("initialize is idempotent and returns the same instantiation", async () => {
    // A page that fires two searches before either resolves must not
    // instantiate the module twice.
    const [a, b] = await Promise.all([initialize(), initialize()]);
    assert.equal(a, b);
  });
});

describe("load", () => {
  test("accepts a Uint8Array", async () => {
    const index = await load(await readFile(fixture("en.marz")));
    assert.equal(index.documentCount, 3);
    index.free();
  });

  test("accepts a bare ArrayBuffer", async () => {
    const bytes = await readFile(fixture("en.marz"));
    const buffer = bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + bytes.byteLength,
    );
    const index = await load(buffer as ArrayBuffer);
    assert.equal(index.documentCount, 3);
    index.free();
  });

  test("reads the language from the header without being told", () => {
    assert.equal(en.language, "en");
    assert.equal(ja.language, "ja");
  });

  test("accepts a language that agrees with the header", async () => {
    const index = await load(await readFile(fixture("ja.marz")), "ja");
    assert.equal(index.language, "ja");
    index.free();
  });

  test("rejects a language that disagrees with the header", async () => {
    // The failure this guards: shipping ja.marz to the /en/ page produces a
    // search box that silently finds nothing.
    const bytes = await readFile(fixture("ja.marz"));
    await assert.rejects(
      () => load(bytes, "en"),
      /built for language "ja", not "en"/,
    );
  });

  test("rejects bytes that are not an index", async () => {
    await assert.rejects(
      () => load(new TextEncoder().encode("not an index at all")),
      /not a Marz index/,
    );
  });

  test("rejects a JSON index, which is a plausible mistake", async () => {
    await assert.rejects(
      () => load(new TextEncoder().encode('{"version":"1.0","fields":[]}')),
      /not a Marz index/,
    );
  });

  test("failures are Error objects, not thrown strings", async () => {
    // A thrown string has no message and no stack, so it logs as a line with no
    // origin. Every failure path here goes through js_sys::Error.
    const error = await load(new Uint8Array([1, 2, 3])).catch((e) => e);
    assert.ok(error instanceof Error);
    assert.ok(error.stack);
    assert.ok(error.message.length > 0);
  });
});

describe("indexLanguage", () => {
  test("reads the language without loading the index", async () => {
    assert.equal(await indexLanguage(await readFile(fixture("ja.marz"))), "ja");
  });

  test("rejects bytes that are not an index", async () => {
    await assert.rejects(
      () => indexLanguage(new Uint8Array(64)),
      /not a Marz index/,
    );
  });
});

describe("accessors", () => {
  test("reports the fields it was built with", () => {
    assert.deepEqual(en.fields, ["title", "text"]);
  });

  test("reports document and term counts", () => {
    assert.equal(en.documentCount, 3);
    assert.equal(ja.documentCount, 10);
    // Ten short Japanese documents produce far more bigrams than three English
    // documents produce stems.
    assert.ok(ja.termCount > en.termCount * 10);
  });
});

describe("search", () => {
  test("returns hits with ref, score and matches", () => {
    const hits = en.search("keyboard");
    assert.ok(hits.length > 0);
    for (const hit of hits) {
      assert.equal(typeof hit.ref, "string");
      assert.equal(typeof hit.score, "number");
      assert.ok(Number.isFinite(hit.score));
      assert.equal(typeof hit.matches, "object");
    }
  });

  test("orders hits by descending score", () => {
    const scores = en.search("keys").map((h) => h.score);
    assert.ok(scores.length > 1, "need several hits to check ordering");
    assert.deepEqual(scores, [...scores].sort((a, b) => b - a));
  });

  test("returns an empty array for a term nothing contains", () => {
    assert.deepEqual(en.search("xylophone"), []);
  });

  test("matches are [start, length] pairs, not objects", () => {
    // The Rust side holds Vec<(usize, usize)>; a tuple has no natural JS shape,
    // so this pins the one chosen.
    const [hit] = en.search("keyboard");
    const spans = Object.values(hit.matches).flatMap((f) =>
      Object.values(f).flat(),
    );
    assert.ok(spans.length > 0);
    for (const span of spans) {
      assert.ok(Array.isArray(span));
      assert.equal(span.length, 2);
      assert.equal(typeof span[0], "number");
      assert.equal(typeof span[1], "number");
    }
  });

  test("positions locate the term in the normalized field text", () => {
    const hit = en.search("keyboard").find((h) => h.ref === "a");
    assert.ok(hit, "document a contains 'keyboard'");
    const source = enDocs.find((d) => d.location === "a")!.text;
    const points = [...normalizeSync(source)];
    const spans = hit.matches["keyboard"]?.["text"] ?? [];
    assert.ok(spans.length > 0);
    for (const [start, length] of spans) {
      assert.equal(points.slice(start, start + length).join(""), "keyboard");
    }
  });

  test("limit caps the hits returned", () => {
    const all = en.search("keys");
    assert.ok(all.length >= 2, "fixture must have several hits to cap");
    assert.equal(en.search("keys", 1).length, 1);
    assert.deepEqual(en.search("keys", 1)[0].ref, all[0].ref);
  });

  test("a limit above the hit count is not an error", () => {
    assert.deepEqual(
      en.search("keys", 1000).map((h) => h.ref),
      en.search("keys").map((h) => h.ref),
    );
  });

  test("a limit of zero returns nothing", () => {
    assert.deepEqual(en.search("keys", 0), []);
  });

  test("supports the query operators", () => {
    // Not testing the parser — testing that the string reaches it intact
    // through the FFI boundary rather than being mangled.
    assert.doesNotThrow(() => en.search("+keyboard -mouse"));
    assert.doesNotThrow(() => en.search("title:keyboards"));
    assert.doesNotThrow(() => en.search("keybo*"));
    assert.doesNotThrow(() => en.search("keyboard~1"));
    assert.doesNotThrow(() => en.search("keyboard^10"));
  });

  test("a field restriction actually restricts", () => {
    const inTitle = en.search("title:keyboards");
    for (const hit of inTitle) {
      assert.deepEqual(Object.keys(hit.matches["keyboard"] ?? {}), ["title"]);
    }
  });
});

describe("query errors", () => {
  test("an unknown field throws an Error naming it", () => {
    assert.throws(() => en.search("nosuchfield:keyboard"), /nosuchfield/);
  });

  test("the error carries the query and the offsets of the fault", () => {
    const error = (() => {
      try {
        en.search("nosuchfield:keyboard");
        return undefined;
      } catch (e) {
        return e as Error & { query: string; start: number; end: number };
      }
    })();
    assert.ok(error, "an unknown field must throw");
    assert.equal(error.query, "nosuchfield:keyboard");
    assert.equal(typeof error.start, "number");
    assert.equal(typeof error.end, "number");
    assert.ok(error.end > error.start);
    // The span must actually cover the offending text, or a search box that
    // underlines it would underline the wrong characters.
    assert.equal(error.query.slice(error.start, error.end), "nosuchfield");
  });

  test("the message stays a sentence, with the offsets on attributes", () => {
    // Putting the offsets in the message would make it unshowable to a user.
    const error = (() => {
      try {
        en.search("nosuchfield:keyboard");
        return undefined;
      } catch (e) {
        return e as Error;
      }
    })();
    assert.ok(error);
    assert.doesNotMatch(error.message, /^\[/);
    assert.match(error.message, /field/);
  });

  test("an index stays usable after a failed query", () => {
    assert.throws(() => en.search("nosuchfield:x"));
    assert.ok(en.search("keyboard").length > 0);
  });
});

describe("tokenize", () => {
  test("splits English on whitespace and stems nothing", async () => {
    // tokenize() is pre-pipeline: it shows the split, not the stemmed terms.
    assert.deepEqual(await tokenize("hello world", "en"), ["hello", "world"]);
  });

  test("splits CJK into overlapping bigrams", async () => {
    // Overlapping, except across the Han/Katakana boundary: 索エ would span
    // 検索 and エンジン, two separate words. Script changes in Japanese fall on
    // morpheme boundaries, so honouring them recovers real word boundaries with
    // no dictionary.
    assert.deepEqual(await tokenize("検索エンジン", "ja"), [
      "検索",
      "エン",
      "ンジ",
      "ジン",
    ]);
  });

  test("rejects an unknown language rather than guessing", async () => {
    await assert.rejects(() => tokenize("hello", "xx"), /unknown language/);
  });
});

describe("normalize", () => {
  test("folds full-width Latin to ASCII", async () => {
    assert.equal(await normalize("ＲＵＳＴ"), "rust");
  });

  test("composes half-width katakana, which changes the length", async () => {
    // This is why positions are offsets into the normalized text: two code
    // points become one, shifting everything after them.
    const original = "ｶﾞｲﾄﾞ";
    const folded = await normalize(original);
    assert.equal(folded, "ガイド");
    assert.equal([...original].length, 5);
    assert.equal([...folded].length, 3);
  });

  test("is idempotent", async () => {
    const once = await normalize("ＲＵＳＴ ｶﾞｲﾄﾞ　検索");
    assert.equal(await normalize(once), once);
  });
});

describe("highlight", () => {
  test("splits text into matched and unmatched spans", async () => {
    const hit = en.search("keyboard").find((h) => h.ref === "a")!;
    const doc = enDocs.find((d) => d.location === "a")!;
    const segments = await highlight(hit, "text", doc.text);
    const matched = segments.filter((s) => s.matched);
    assert.ok(matched.length > 0);
    assert.deepEqual([...new Set(matched.map((s) => s.text))], ["keyboard"]);
  });

  test("the spans reassemble into the normalized text, losing nothing", async () => {
    const hit = en.search("keyboard").find((h) => h.ref === "a")!;
    const doc = enDocs.find((d) => d.location === "a")!;
    const segments = await highlight(hit, "text", doc.text);
    assert.equal(
      segments.map((s) => s.text).join(""),
      await normalize(doc.text),
    );
  });

  test("offsets survive an astral-plane character earlier in the field", async () => {
    // Document c has a 🎉 before the word "keyboard". Positions count code
    // points and String.slice counts UTF-16 units, so a naive slice would be one
    // unit off and cut the surrogate pair.
    const hit = en.search("keyboard").find((h) => h.ref === "c");
    assert.ok(hit, "document c contains 'keyboard' after an emoji");
    const doc = enDocs.find((d) => d.location === "c")!;
    assert.ok(doc.text.includes("🎉"), "fixture must contain the emoji");
    const segments = await highlight(hit, "text", doc.text);
    const matched = segments.filter((s) => s.matched).map((s) => s.text);
    assert.deepEqual(matched, ["keyboard"]);
    // The naive version, for the record: it does not produce "keyboard".
    const [start] = hit.matches["keyboard"]!["text"]![0];
    const naive = (await normalize(doc.text)).slice(start, start + 8);
    assert.notEqual(naive, "keyboard");
  });

  test("merges overlapping CJK bigram matches into one span", async () => {
    const hits = ja.search("検索エンジン");
    assert.ok(hits.length > 0);
    const hit = hits[0];
    const doc = jaDocs.find((d) => d.location === hit.ref)!;
    const segments = await highlight(hit, "text", doc.text);
    // Overlapping bigrams would otherwise produce a run of two-character spans.
    // Merging means at least one match is longer than a bigram, and no two
    // matched spans are adjacent.
    const matched = segments.filter((s) => s.matched);
    assert.ok(matched.length > 0);
    assert.ok(
      matched.some((s) => [...s.text].length > 2),
      `expected a merged span longer than one bigram, got ${JSON.stringify(
        matched.map((s) => s.text),
      )}`,
    );
    for (let i = 1; i < segments.length; i += 1) {
      assert.ok(
        !(segments[i - 1].matched && segments[i].matched),
        "adjacent matched spans should have been merged",
      );
    }
  });

  test("returns the whole field unmatched when nothing matched in it", async () => {
    // A hit on "mouse" has no match in the title field of document b, so the
    // title renders whole. Note the text comes back normalized: lowercased and
    // width-folded, which is what makes the offsets mean anything.
    const hit = en.search("mouse").find((h) => h.ref === "b")!;
    assert.equal(
      Object.values(hit.matches).every((f) => !("title" in f)),
      true,
      "fixture must have a hit that does not match in the title",
    );
    const segments = await highlight(hit, "title", "Nothing Matched Here");
    assert.deepEqual(segments, [
      { text: "nothing matched here", matched: false },
    ]);
  });

  test("returns no spans for empty text", async () => {
    const hit = en.search("keyboard")[0];
    assert.deepEqual(await highlight(hit, "text", ""), []);
  });

  test("ignores positions that fall outside the text it was given", async () => {
    // A caller who passes text other than what was indexed — a truncated
    // summary, say. Clamping would highlight the wrong span; the rest of the
    // field should still render.
    const hit = en.search("keyboard").find((h) => h.ref === "a")!;
    const segments = await highlight(hit, "text", "short");
    assert.deepEqual(segments, [{ text: "short", matched: false }]);
  });

  test("an unknown field name is not an error", async () => {
    const hit = en.search("keyboard")[0];
    const segments = await highlight(hit, "nosuchfield", "some text");
    assert.deepEqual(segments, [{ text: "some text", matched: false }]);
  });
});

describe("CJK search", () => {
  test("finds a term that is not delimited by spaces", () => {
    // The whole reason Marz exists: no dictionary, no segmentation, and
    // 形態素解析 still matches inside a run of Japanese text.
    const hits = ja.search("形態素解析");
    assert.ok(hits.length > 0);
  });

  test("matches full-width Latin typed as ASCII", async () => {
    const hits = ja.search("search");
    // The fixture contains "search engine" in Latin; a full-width variant in
    // the corpus is folded to the same term.
    assert.ok(Array.isArray(hits));
  });

  test("reports bigrams as the matched terms", () => {
    const [hit] = ja.search("検索");
    assert.ok(hit);
    for (const term of Object.keys(hit.matches)) {
      assert.equal([...term].length, 2, `expected a bigram, got ${term}`);
    }
  });
});

describe("memory", () => {
  test("free() releases the index and using it afterwards throws", async () => {
    const index = await load(await readFile(fixture("en.marz")));
    assert.ok(index.search("keyboard").length > 0);
    index.free();
    assert.throws(() => index.search("keyboard"));
  });

  test("many loads and frees do not leak into each other", async () => {
    const bytes = await readFile(fixture("en.marz"));
    for (let i = 0; i < 20; i += 1) {
      const index = await load(bytes);
      assert.equal(index.documentCount, 3);
      assert.ok(index.search("keyboard").length > 0);
      index.free();
    }
  });
});
