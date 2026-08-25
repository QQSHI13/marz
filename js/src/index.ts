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
  ready ??= source === undefined
    ? init()
    : init({ module_or_path: source });
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
): Promise<MarzIndex> {
  await initialize();

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
      throw new Error(
        `could not fetch index from ${String(source)}: ` +
          `${response.status} ${response.statusText}`,
      );
    }
    bytes = new Uint8Array(await response.arrayBuffer());
  }

  return MarzIndex.load(bytes, expectedLanguage);
}

/** Language codes this build supports. */
export async function languages(): Promise<string[]> {
  await initialize();
  return wasmLanguages();
}

/** The version of Marz this module was built from. */
export async function version(): Promise<string> {
  await initialize();
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
): Promise<string> {
  await initialize();
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
 */
export async function tokenize(
  text: string,
  language: string,
): Promise<string[]> {
  await initialize();
  return wasmTokenize(text, language);
}

/**
 * Apply the normalization the indexer applies before tokenizing.
 *
 * Match positions are offsets into this string, not into the input. See
 * {@link highlight}.
 */
export async function normalize(text: string): Promise<string> {
  await initialize();
  return wasmNormalize(text);
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
): Promise<Segment[]> {
  await initialize();
  const normalized = wasmNormalize(text);
  // Code points, so that offsets line up on text containing astral-plane
  // characters. `[...string]` iterates code points; indexing does not.
  const points = [...normalized];

  const spans: Array<[number, number]> = [];
  for (const fields of Object.values(hit.matches)) {
    for (const [start, length] of fields[field] ?? []) {
      // A position past the end means the text passed in is not the text that
      // was indexed. Clamping would silently highlight the wrong span, so skip
      // it and leave the rest of the field readable.
      if (start >= 0 && start + length <= points.length) {
        spans.push([start, start + length]);
      }
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
