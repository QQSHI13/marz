/**
 * Tests for client-side incremental updates (`MarzBuilder.fromIndex`/`remove`).
 *
 * These exercise the boundary, not the engine: that a rehydrated builder
 * restores its fields, that `remove` reports presence, that staged additions
 * meet removals in the rebuilt index. Engine behaviour is covered by the Rust
 * suite; ranking assertions here would fail for reasons outside this layer.
 *
 * Needs a WebAssembly build with the `builder` feature
 * (`scripts/build-wasm.sh --features builder`). Against the default
 * search-only build every test skips: there is no `MarzBuilder` export to
 * test, and failing would blame the caller for shipping the smaller module.
 */

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { before, describe, test } from "node:test";

import {
  builderFromIndex,
  createBuilder,
  initialize,
  load,
} from "../src/index.ts";

const fixture = (name: string) =>
  fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url));

/** Whether this WebAssembly build includes the client-side builder. */
async function hasBuilder(): Promise<boolean> {
  const pkg = (await import("../pkg/marz_wasm.js")) as unknown as {
    MarzBuilder?: unknown;
  };
  return pkg.MarzBuilder !== undefined;
}

before(async () => {
  // Same bytes-passing rule as the search tests: Node's `fetch` does not
  // implement `file:` URLs, so path resolution fails there.
  await initialize(await readFile(fixture("../pkg/marz_wasm_bg.wasm")));
});

describe("incremental updates", () => {
  test("fromIndex restores fields and rebuilds an equivalent index", async (t) => {
    if (!(await hasBuilder())) {
      t.skip("search-only build has no MarzBuilder");
      return;
    }
    const index = await load(await readFile(fixture("en.marz")));
    try {
      const builder = await builderFromIndex(index);
      try {
        assert.deepEqual(builder.fields, ["title", "text"]);
        assert.equal(builder.language, "en");
        assert.equal(builder.staged, 0);
        const rebuilt = builder.buildAndLoad();
        try {
          assert.equal(rebuilt.documentCount, 3);
          assert.ok(rebuilt.search("keyboard").length > 0);
        } finally {
          rebuilt.free();
        }
      } finally {
        builder.free();
      }
    } finally {
      index.free();
    }
  });

  test("remove deletes and reports presence", async (t) => {
    if (!(await hasBuilder())) {
      t.skip("search-only build has no MarzBuilder");
      return;
    }
    const index = await load(await readFile(fixture("en.marz")));
    try {
      const builder = await builderFromIndex(index);
      try {
        assert.equal(builder.remove("c"), true);
        // Unknown references are a no-op, not an error: bulk syncs diff
        // against stale ids.
        assert.equal(builder.remove("missing"), false);
        assert.equal(builder.remove("c"), false);
        const rebuilt = builder.buildAndLoad();
        try {
          assert.equal(rebuilt.documentCount, 2);
          // Document a still mentions keyboards; only c is gone.
          const refs = rebuilt.search("keyboard").map((h) => h.ref);
          assert.ok(refs.includes("a"));
          assert.ok(!refs.includes("c"));
        } finally {
          rebuilt.free();
        }
      } finally {
        builder.free();
      }
    } finally {
      index.free();
    }
  });

  test("remove on a fresh builder throws", async (t) => {
    if (!(await hasBuilder())) {
      t.skip("search-only build has no MarzBuilder");
      return;
    }
    const builder = await createBuilder("en");
    try {
      builder.field("body");
      builder.add({ id: "a", body: "hello" });
      assert.throws(() => builder.remove("a"), /fromIndex/);
    } finally {
      builder.free();
    }
  });

  test("a staged add then removed stays removed", async (t) => {
    if (!(await hasBuilder())) {
      t.skip("search-only build has no MarzBuilder");
      return;
    }
    const index = await load(await readFile(fixture("en.marz")));
    try {
      const builder = await builderFromIndex(index);
      try {
        builder.add({ id: "d", title: "new", text: "keyboard" });
        assert.equal(builder.remove("d"), true);
        const rebuilt = builder.buildAndLoad();
        try {
          assert.equal(rebuilt.documentCount, 3);
        } finally {
          rebuilt.free();
        }
      } finally {
        builder.free();
      }
    } finally {
      index.free();
    }
  });

  test("upsert through a rehydrated builder replaces", async (t) => {
    if (!(await hasBuilder())) {
      t.skip("search-only build has no MarzBuilder");
      return;
    }
    const index = await load(await readFile(fixture("en.marz")));
    try {
      const builder = await builderFromIndex(index);
      try {
        // Document c mentions keyboards; re-adding it without that term
        // must drop its old postings without growing the count. Document a
        // still matches, so the query is not empty — only c is gone.
        const before = builder.buildAndLoad();
        const hadC = before
          .search("keyboard")
          .some((h) => h.ref === "c");
        before.free();
        assert.ok(hadC, "fixture document c must mention keyboards");
        builder.add({ id: "c", title: "unrelated", text: "nothing here" });
        const rebuilt = builder.buildAndLoad();
        try {
          assert.equal(rebuilt.documentCount, 3);
          const refs = rebuilt.search("keyboard").map((h) => h.ref);
          assert.ok(refs.includes("a"));
          assert.ok(!refs.includes("c"));
        } finally {
          rebuilt.free();
        }
      } finally {
        builder.free();
      }
    } finally {
      index.free();
    }
  });

  test("fields cannot be declared twice after rehydration", async (t) => {
    if (!(await hasBuilder())) {
      t.skip("search-only build has no MarzBuilder");
      return;
    }
    const index = await load(await readFile(fixture("en.marz")));
    try {
      const builder = await builderFromIndex(index);
      try {
        assert.throws(() => builder.field("title"), /already declared/);
      } finally {
        builder.free();
      }
    } finally {
      index.free();
    }
  });

  test("building twice gives two equivalent indexes", async (t) => {
    if (!(await hasBuilder())) {
      t.skip("search-only build has no MarzBuilder");
      return;
    }
    const index = await load(await readFile(fixture("en.marz")));
    try {
      const builder = await builderFromIndex(index);
      try {
        assert.equal(builder.remove("c"), true);
        const first = builder.buildAndLoad();
        const second = builder.buildAndLoad();
        try {
          assert.equal(first.documentCount, 2);
          assert.equal(second.documentCount, 2);
          assert.deepEqual(
            first.search("keyboard").map((h) => h.ref),
            second.search("keyboard").map((h) => h.ref),
          );
        } finally {
          first.free();
          second.free();
        }
      } finally {
        builder.free();
      }
    } finally {
      index.free();
    }
  });
});
