# Marz

Offline search engine for [docsforge](https://github.com/QQSHI13/docsforge). Replaces `lunr` + `lunr-languages` + `jieba` with a single Rust core exposed through Python (build-time indexing) and WebAssembly (browser search).

## Objectives

- Same scoring behavior as lunr.js / lunr.py
- As fast as possible
- Complete offline support
- Lightweight bundles and indexes
- First-class CJK support

## Architecture

- `crates/marz-core` — Rust search engine (indexing + querying)
- `python/` — PyO3 bindings for docsforge build step
- `crates/marz-wasm` + `js/` — wasm-bindgen + TypeScript for browser

## Status

Phase 0 bootstrap in progress.

## License

GPL-3.0
