#!/usr/bin/env bash
# Build the shipped WebAssembly module: wasm-pack, then wasm-opt.
#
# Two reasons this is a script rather than a one-line wasm-pack invocation.
#
# First, wasm-opt's default baseline is older than the WebAssembly rustc now
# emits. LLVM lowers large copies to `memory.copy` (bulk memory) and float→int
# casts to `i32.trunc_sat_f64_u` (non-trapping float-to-int), and wasm-opt
# rejects the module rather than passing it through. Both features shipped in
# every major browser by 2020 — below what wasm-bindgen's own output already
# requires — so enabling them costs no reach.
#
# Second, when wasm-pack's own wasm-opt call fails it prints the error, carries
# on, and ships the *unoptimized* module. The build still "succeeds" and the
# result still works, so the only symptom is 191 KB where 175 KB was expected.
# Running the optimizer here, with `set -e`, makes that failure stop the build.
#
# wasm-pack 0.15 only reads its `wasm-opt` metadata for the built-in
# `--release`/`--profiling` profiles, not for the custom `wasm-release` profile
# this project builds with — hence `--no-opt` plus an explicit step. The metadata
# in crates/marz-wasm/Cargo.toml covers anyone who builds with `--release`.
#
# Usage: scripts/build-wasm.sh [extra cargo args...]
#   scripts/build-wasm.sh                      # search only, what the site ships
#   scripts/build-wasm.sh --features builder   # plus the client-side index builder
set -euo pipefail

cd "$(dirname "$0")/.."

wasm_pack_args=(
    build crates/marz-wasm
    --out-dir ../../js/pkg
    --target web
    --profile wasm-release
    --no-opt
)
if [ "$#" -gt 0 ]; then
    wasm_pack_args+=(-- "$@")
fi
wasm-pack "${wasm_pack_args[@]}"

# wasm-pack downloads a wasm-opt into ~/.cache/.wasm-pack rather than requiring
# one on PATH, so prefer a PATH binary and fall back to that cache.
opt=$(command -v wasm-opt || find "$HOME/.cache/.wasm-pack" -name wasm-opt -type f 2>/dev/null | head -1)
if [ -z "$opt" ]; then
    echo "wasm-opt not found on PATH or in ~/.cache/.wasm-pack" >&2
    exit 1
fi

"$opt" js/pkg/marz_wasm_bg.wasm -o js/pkg/marz_wasm_bg.wasm \
    -Oz --enable-bulk-memory --enable-nontrapping-float-to-int

# wasm-pack writes a `.gitignore` containing `*` into its output directory, on the
# assumption that the directory is a standalone package it will publish itself.
# Here the directory is a subdirectory of the js package, and npm reads nested
# `.gitignore` files as `.npmignore` — so that one line silently excludes the
# entire WebAssembly module from `npm pack`, and `files` in package.json cannot
# override it. The symptom is a published tarball with the wrapper and no engine.
# js/pkg is ignored at the repository root, so nothing needs this file.
rm -f js/pkg/.gitignore
