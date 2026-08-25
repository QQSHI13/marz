set fallback := true

# Default recipe: check everything
default:
    just check

# Check Rust workspace
check:
    cargo check --workspace

# Run Rust tests
test:
    cargo test --workspace

# Run every test suite
test-all: test test-python test-js

# Format Rust code
fmt:
    cargo fmt --all

# Lint Rust code, including tests, examples and the optional wasm builder
lint:
    cargo clippy --workspace --all-targets -- -D warnings
    cargo clippy -p marz-wasm --target wasm32-unknown-unknown --features builder -- -D warnings

# Build Python wheel (requires maturin)
build-python:
    cd python && maturin build --release

# Run the Python binding tests. `python -m pytest` rather than bare `pytest` so
# this uses the interpreter maturin just installed the extension into, instead of
# whichever pytest happens to be first on PATH.
test-python:
    cd python && maturin develop && python -m pytest -q

# Build the shipped WASM package. See scripts/build-wasm.sh for why wasm-opt runs
# as a separate step with explicit feature flags — the short version is that
# wasm-pack ships an unoptimized module on wasm-opt failure without failing.
build-wasm:
    scripts/build-wasm.sh

# Build WASM with the client-side index builder included
build-wasm-builder:
    scripts/build-wasm.sh --features builder

# Build JS wrapper (requires npm install in js/)
build-js:
    cd js && npm run build

# Run the JS binding tests. Node strips the TypeScript at runtime, so this needs
# no dependencies beyond the WebAssembly build the tests load.
test-js: build-wasm
    cd js && node --test 'test/*.test.ts'

# Regenerate the JS test fixtures from their corpora.
fixtures:
    cargo run --example mkfixture -p marz-core -- \
        js/fixtures/corpus_en.json js/fixtures/en.marz en
    cargo run --example mkfixture -p marz-core -- \
        js/fixtures/corpus_ja.json js/fixtures/ja.marz ja

# Regenerate the English ranking oracle in tests/fixtures from lunr.js.
#
# Rarely needed, and never to make a failing test pass: a diff here means either
# the corpus changed or Marz's ranking did, and only the first is a reason to
# regenerate. lunr.js — not lunr.py — is the oracle; see
# crates/marz-core/tests/golden.rs for why the two cannot both be matched.
golden:
    cd tests && npm install && node generate_golden.js

# Report the shipped size of the WebAssembly module, which is what a page pays.
# `cargo build`'s artifact is roughly three times this: wasm-bindgen's gc pass
# strips the unreachable half, so measuring the cargo output overstates
# everything and misattributes where the bytes went. CI asserts a ceiling on this
# number; see the `wasm` job in .github/workflows/ci.yml.
size: build-wasm
    @ls -l js/pkg/marz_wasm_bg.wasm | awk '{print $5 " bytes raw"}'
    @gzip -9 -c js/pkg/marz_wasm_bg.wasm | wc -c | awk '{print $1 " bytes gzipped"}'

# Build all bindings
build-all: build-python build-wasm build-js

# Inspect what each registry would receive, without publishing anything.
#
# The npm check is the one that has caught a real bug: the tarball shipped
# without its WebAssembly, because wasm-pack leaves a `.gitignore` containing `*`
# in its output directory and npm honours a nested one over `files`. Expect ~8
# files and ~98 kB. Five files and 6 kB means the engine is missing.
publish-dry: build-wasm
    cd js && npm install && npm run build:ts && npm pack --dry-run
    cargo publish -p marz-core --dry-run
    cd python && maturin sdist -o ../target/sdist
    @echo
    @echo "Nothing was published. See RELEASING.md to cut a real release."
