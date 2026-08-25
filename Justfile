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

# Run the Python binding tests (requires maturin and pytest)
test-python:
    cd python && maturin develop && pytest -q

# Build WASM package (requires wasm-pack)
build-wasm:
    wasm-pack build crates/marz-wasm --out-dir ../../js/pkg --target web --profile wasm-release

# Build WASM with the client-side index builder included
build-wasm-builder:
    wasm-pack build crates/marz-wasm --out-dir ../../js/pkg --target web --profile wasm-release -- --features builder

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

# Report the shipped size of the WebAssembly module, which is what a page pays.
# `cargo build`'s artifact is roughly three times this: wasm-bindgen's gc pass
# strips the unreachable half, so measuring the cargo output overstates
# everything and misattributes where the bytes went.
size: build-wasm
    @ls -l js/pkg/marz_wasm_bg.wasm | awk '{print $5 " bytes raw"}'
    @gzip -9 -c js/pkg/marz_wasm_bg.wasm | wc -c | awk '{print $1 " bytes gzipped"}'

# Build all bindings
build-all: build-python build-wasm build-js
