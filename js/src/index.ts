/**
 * Marz — offline search with first-class CJK support.
 *
 * This package is the browser half. An index is built ahead of time and shipped
 * as bytes; this loads them and searches.
 *
 * ```ts
 * import { load, highlight } from "marz-search";
 *
 * const index = await load("/search/ja.marz");
 * for (const hit of index.search("検索エンジン", 10)) {
 *   console.log(hit.ref, hit.score);
 * }
 * ```
 *
 * The WebAssembly module is initialized on the first call and reused after, so
 * nothing here needs an explicit setup step.
 */

import init, {
  MarzIndex,
  indexLanguage as wasmIndexLanguage,
  languages as wasmLanguages,
  normalize as wasmNormalize,
  tokenize as wasmTokenize,
  version as wasmVersion,
} from "../pkg/marz_wasm.js";

export type { Matches, SearchResult } from "../pkg/marz_wasm.js";
export { MarzIndex } from "../pkg/marz_wasm.js";

import type { InitInput, SearchResult } from "../pkg/marz_wasm.js";

/**
 * Where to find the `.wasm`, when it is not resolvable beside the generated JS.
 *
 * A URL, a `Response`, or the bytes themselves. Needed under Node, and in
 * browsers whose bundler rewrites asset paths.
 */
export type WasmSource = InitInput | Promise<InitInput>;

/**
 * The in-flight or completed initialization.
 *
 * Held as the promise rather than a boolean so that concurrent first calls —
 * a page that fires two searches before either resolves — await one
 * instantiation instead of racing two.
 */
let ready: Promise<unknown> | undefined;

/**
 * Initialize the WebAssembly module.
 *
 * Optional: every function here calls it. Call it directly to control *when* the
 * module downloads — during idle time after first paint, say, rather than when
 * the user first types.
 *
 * In a browser the argument can be omitted; wasm-pack resolves the `.wasm`
 * beside the generated JavaScript. Under Node — tests, or server-side rendering
 * — pass the bytes, because Node's `fetch` does not implement `file:` URLs and a
 * path would fail with a network error that names nothing useful:
 *
 * ```ts
 * import { readFile } from "node:fs/promises";
 * await initialize(await readFile("node_modules/marz-search/pkg/marz_wasm_bg.wasm"));
 * ```
 */
export function initialize(source?: WasmSource): Promise<unknown> {
  // The object form: wasm-bindgen deprecated passing the source positionally
  // and warns on every call. Omit the argument entirely when there is none, so
  // that a browser still gets the default resolution.
  //
  // `??=` would cache a rejection forever and ignore a later source (e.g. a
  // second `initialize(bytes)` after a bare `initialize()`). Assign only when
  // uninitialized, and clear on failure so the next call retries.
  if (!ready) {
    const p =
      source === undefined ? init() : init({ module_or_path: source });
    ready = p.catch((e) => {
      ready = undefined;
      throw e;
    });
  }
  return ready;
}

/**
 * Fetch and load an index.
 *
 * `source` is a URL to fetch, or bytes already in hand — from a cache, a
 * `File` the user dropped, or a worker `postMessage`.
 *
 * Give `expectedLanguage` to assert what the bytes should be. A pipeline that
 * ships the wrong per-locale file otherwise produces a search box that finds
 * nothing, with nothing anywhere to explain it.
 *
 * The index holds WebAssembly memory. Keep it for the lifetime of the page —
 * the normal case — or call `.free()` when done.
 */
export async function load(
  source: string | URL | Request | ArrayBuffer | Uint8Array,
  expectedLanguage?: string,
  wasmSource?: WasmSource,
): Promise<MarzIndex> {
  await initialize(wasmSource);

  let bytes: Uint8Array;
  if (source instanceof Uint8Array) {
    bytes = source;
  } else if (source instanceof ArrayBuffer) {
    bytes = new Uint8Array(source);
  } else {
    const response = await fetch(source as string | URL | Request);
    if (!response.ok) {
      // The status is the whole diagnosis here: a 404 means the build did not
      // emit the index, a 200 serving HTML means a dev server rewrote the path
      // to index.html, and both look identical from inside the reader.
      // `Request` stringifies to `[object Request]`, so quote its URL instead.
      const where =
        typeof source !== "string" &&
        typeof (source as Request).url === "string"
          ? (source as Request).url
          : String(source);
      throw new Error(
        `could not fetch index from ${where}: ` +
          `${response.status} ${response.statusText}`,
      );
    }
    bytes = new Uint8Array(await response.arrayBuffer());
  }

  return MarzIndex.load(bytes, expectedLanguage);
}

/** Language codes this build supports. */
export async function languages(
  wasmSource?: WasmSource,
): Promise<string[]> {
  await initialize(wasmSource);
  return wasmLanguages();
}

/** The version of Marz this module was built from. */
export async function version(
  wasmSource?: WasmSource,
): Promise<string> {
  await initialize(wasmSource);
  return wasmVersion();
}

/**
 * Report the language an index was built for, without loading it.
 *
 * Reads the 64-byte header and stops, so it is cheap enough to run against a
 * fetched buffer before deciding whether to load it.
 */
export async function indexLanguage(
  bytes: ArrayBuffer | Uint8Array,
  wasmSource?: WasmSource,
): Promise<string> {
  await initialize(wasmSource);
  return wasmIndexLanguage(
    bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes),
  );
}

/**
 * Split text the way the index would.
 *
 * The fastest way to understand a CJK result. `tokenize("検索エンジン", "ja")`
 * returns `["検索", "エン", "ンジ", "ジン"]`: overlapping bigrams, which is why a
 * query for `検索` matches a document that never contains that word delimited by
 * spaces. No bigram crosses the Han/Katakana boundary — `索エ` would span two
 * words — so script changes recover real word boundaries for free.
 *
 * Async only to cover WASM initialization; after `initialize()` the work itself
 * is synchronous (unlike `MarzIndex.search`, which is sync). The `await` is
 * still required on first use under Node/SSR.
 */
export async function tokenize(
  text: string,
  language: string,
  wasmSource?: WasmSource,
): Promise<string[]> {
  await initialize(wasmSource);
  return wasmTokenize(text, language);
}

/**
 * Apply the normalization the indexer applies before tokenizing.
 *
 * Match positions are offsets into this string, not into the input. See
 * {@link highlight}. Async only for initialization — see {@link tokenize}.
 *
 * Pass the index's language for Turkish (`"tr"`), whose `I` folds to `ı`;
 * every other language folds the default way. A multi-language code containing
 * Turkish folds the default way instead — no single folding serves both
 * members, so dotted-capital-I offsets in Turkish text may shift by a
 * character; search itself is unaffected.
 */
export async function normalize(
  text: string,
  language?: string,
  wasmSource?: WasmSource,
): Promise<string> {
  await initialize(wasmSource);
  return wasmNormalize(text, language);
}

/** A span of field text, flagged as matched or not. */
export interface Segment {
  /** The text of this span. */
  text: string;
  /** Whether a query term matched here. */
  matched: boolean;
}

/**
 * Split field text into matched and unmatched spans, ready to render.
 *
 * Positions in a {@link SearchResult} are hard to use correctly by hand, for two
 * reasons that both produce output that looks right on English and silently
 * corrupts everything else:
 *
 * - They are offsets into the *normalized* text, and normalization is not
 *   length-preserving: `ｶﾞ` is two code points that become one `ガ`. Every
 *   offset after such a character is shifted relative to the original.
 * - They count Unicode code points, while `String.prototype.slice` counts UTF-16
 *   code units. One emoji earlier in the field puts every subsequent span one
 *   unit off, which slices a surrogate pair in half.
 *
 * So this normalizes the text and indexes it as an array of code points. The
 * returned spans therefore quote *normalized* text — lowercased, with widths
 * folded — which is what makes the offsets mean anything, and is usually what a
 * search result should show anyway.
 *
 * CJK bigrams overlap, so adjacent matches are merged: a hit on `エン` and `ンジ`
 * becomes one span, not two overlapping ones.
 *
 * Pass the index's language for Turkish (`"tr"`), whose normalization folds
 * differently — see {@link normalize}. A multi-language code containing
 * Turkish folds the default way instead, so dotted-capital-I offsets in
 * Turkish text may shift by a character; search itself is unaffected.
 *
 * ```ts
 * const segments = await highlight(hit, "body", doc.body);
 * el.replaceChildren(...segments.map((s) => {
 *   if (!s.matched) return document.createTextNode(s.text);
 *   const mark = document.createElement("mark");
 *   mark.textContent = s.text;
 *   return mark;
 * }));
 * ```
 *
 * Building DOM nodes rather than an HTML string is deliberate: interpolating
 * document text into `innerHTML` is how a search result page becomes an XSS
 * sink.
 */
export async function highlight(
  hit: SearchResult,
  field: string,
  text: string,
  language?: string,
  wasmSource?: WasmSource,
): Promise<Segment[]> {
  await initialize(wasmSource);
  if (!hit || typeof (hit as SearchResult).matches !== "object" || (hit as SearchResult).matches === null) {
    throw new Error("highlight: hit.matches must be an object");
  }
  if (typeof text !== "string" || typeof field !== "string") {
    throw new Error("highlight: field and text must be strings");
  }
  // Same language rule as normalize(): Turkish folds differently, and the
  // offsets only line up when both sides fold the same way.
  const normalized = wasmNormalize(text, language);
  // Code points, so that offsets line up on text containing astral-plane
  // characters. `[...string]` iterates code points; indexing does not.
  const points = [...normalized];

  const spans: Array<[number, number]> = [];
  for (const fields of Object.values(hit.matches)) {
    const spansForField: unknown = fields[field] ?? [];
    // Caller-built hits may hold anything here: a non-array (or a pair that
    // is not a [start, length] pair) would throw in the loop below, and
    // search-result rendering must never throw on shape.
    if (!Array.isArray(spansForField)) {
      continue;
    }
    for (const pair of spansForField) {
      if (!Array.isArray(pair) || pair.length !== 2) {
        continue;
      }
      const [start, length] = pair;
      // Positions are code-point integers from the engine, but a caller-built
      // hit may hold anything: fractional or non-finite spans would silently
      // highlight the wrong slice (`slice` coerces), so skip them like
      // out-of-range ones.
      if (
        !Number.isInteger(start) ||
        !Number.isInteger(length) ||
        start < 0 ||
        length < 0 ||
        start + length > points.length
      ) {
        continue;
      }
      // A position past the end means the text passed in is not the text that
      // was indexed. Clamping would silently highlight the wrong span, so skip
      // it and leave the rest of the field readable.
      spans.push([start, start + length]);
    }
  }

  if (spans.length === 0) {
    return normalized.length > 0
      ? [{ text: normalized, matched: false }]
      : [];
  }

  spans.sort((a, b) => a[0] - b[0] || a[1] - b[1]);

  const merged: Array<[number, number]> = [];
  for (const [start, end] of spans) {
    const last = merged[merged.length - 1];
    // `>=` rather than `>`: bigram matches abut as well as overlap, and two
    // touching spans should render as one highlight.
    if (last && start <= last[1]) {
      last[1] = Math.max(last[1], end);
    } else {
      merged.push([start, end]);
    }
  }

  const segments: Segment[] = [];
  let cursor = 0;
  for (const [start, end] of merged) {
    if (start > cursor) {
      segments.push({
        text: points.slice(cursor, start).join(""),
        matched: false,
      });
    }
    segments.push({ text: points.slice(start, end).join(""), matched: true });
    cursor = end;
  }
  if (cursor < points.length) {
    segments.push({ text: points.slice(cursor).join(""), matched: false });
  }
  return segments;
}

/**
 * A client-side index builder, for content that only exists in the browser.
 *
 * This is the TypeScript shape of `MarzBuilder` from the WebAssembly module —
 * written out here rather than imported from it, so that importing this module
 * never requires a WebAssembly build with the `builder` feature. A search-only
 * build (the default, and what `build-wasm.sh` produces without flags) has no
 * `MarzBuilder` at all; the helpers below throw a plain `Error` saying so
 * instead of failing on a missing export.
 *
 * The builder adds ~39 KB of WebAssembly for tokenization, scoring and
 * serialization that a search box never runs. Ship the search-only module and
 * reach for this only when the documents live client-side — a note-taking app
 * whose notes are in IndexedDB, or a viewer indexing a file the user just
 * dropped on the page.
 */
export interface ClientBuilder {
  /** How many documents are staged and ready to build. */
  readonly staged: number;
  /** The declared field names, in declaration order. */
  readonly fields: string[];
  /** The configured reference field. */
  readonly refField: string;
  /** The configured language code. */
  readonly language: string;
  /** Declare a searchable field. `boost` multiplies the score of matches in it. */
  field(name: string, boost?: number): void;
  /** Stage a document for indexing. Same reference twice replaces (upsert). */
  add(doc: unknown, boost?: number): void;
  /** Stage every document in an iterable. Equivalent to `add` in a loop. */
  addMany(docs: unknown, boost?: number): void;
  /**
   * Remove the document with reference `docRef`.
   *
   * Returns `true` when a document was present and removed, `false` when no
   * document used that reference. Only a builder from {@link builderFromIndex}
   * can remove; a fresh builder throws.
   */
  remove(docRef: string): boolean;
  /** Tokenize and score the staged documents, returning the binary index. */
  build(positions?: boolean): Uint8Array;
  /** Build and load in one step, skipping the serialize/parse round trip. */
  buildAndLoad(): MarzIndex;
  /** Discard the staged documents, keeping the field configuration. */
  clear(): void;
  /** Release the WebAssembly memory held by the builder. */
  free(): void;
}

/** The constructor behind {@link ClientBuilder}, with its static rehydrator. */
type BuilderConstructor = {
  new (
    language: string,
    refField?: string,
    k1?: number,
    b?: number,
  ): ClientBuilder;
  fromIndex(index: MarzIndex, refField?: string): ClientBuilder;
};

/**
 * Resolve the builder constructor, initializing the module first.
 *
 * Separated out because both helpers below need the same two steps — ensure
 * the module is instantiated, then find `MarzBuilder` in it — and the same
 * failure: a search-only build has no such export, which means the page
 * shipped the wrong `.wasm`, not that the caller passed a bad argument.
 */
async function builderConstructor(
  wasmSource?: WasmSource,
): Promise<BuilderConstructor> {
  await initialize(wasmSource);
  // Dynamic import rather than a static one: a static `import { MarzBuilder }`
  // would fail to link against a search-only build that has no such export,
  // breaking every search page to type the few that build client-side.
  const pkg = (await import("../pkg/marz_wasm.js")) as unknown as {
    MarzBuilder?: BuilderConstructor;
  };
  const Ctor = pkg.MarzBuilder;
  if (!Ctor) {
    throw new Error(
      "marz: this WebAssembly build has no MarzBuilder " +
        "(rebuild with scripts/build-wasm.sh --features builder)",
    );
  }
  return Ctor;
}

/**
 * Create a builder for `language`, for documents that only exist client-side.
 *
 * Async only to cover WASM initialization; after `initialize()` the work
 * itself is synchronous. `refField` names the property holding each
 * document's identity and defaults to `"id"`. `k1` and `b` are the BM25
 * tuning parameters (defaults 1.2 and 0.75).
 *
 * ```ts
 * const builder = await createBuilder("ja", "location");
 * builder.field("title", 10.0);
 * builder.field("text");
 * builder.add({ location: "guide/intro", title: "入門", text: "…" });
 * const index = builder.buildAndLoad();
 * ```
 */
export async function createBuilder(
  language: string,
  refField?: string,
  k1?: number,
  b?: number,
  wasmSource?: WasmSource,
): Promise<ClientBuilder> {
  const Ctor = await builderConstructor(wasmSource);
  return new Ctor(language, refField, k1, b);
}

/**
 * Rehydrate a builder from a loaded index for incremental updates.
 *
 * Restores the declared fields (names and boosts) and the language, so
 * documents can be added or removed without re-adding the whole corpus. The
 * reference field name is not stored in the index, so pass it again here when
 * it is not `"id"`. `k1`/`b` come along with the index; there is nothing to
 * set. Building clones the restored state and applies the staged documents on
 * top, so building twice gives two equivalent indexes.
 *
 * ```ts
 * const builder = await builderFromIndex(index, "location");
 * builder.remove("guide/old");
 * builder.add({ location: "guide/new", title: "…", text: "…" });
 * const updated = builder.buildAndLoad();
 * ```
 */
export async function builderFromIndex(
  index: MarzIndex,
  refField?: string,
  wasmSource?: WasmSource,
): Promise<ClientBuilder> {
  const Ctor = await builderConstructor(wasmSource);
  return Ctor.fromIndex(index, refField);
}
