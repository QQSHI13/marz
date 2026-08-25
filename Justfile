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
test-all: test test-python

# Format Rust code
fmt:
    cargo fmt --all

# Lint Rust code
lint:
    cargo clippy --workspace -- -D warnings

# Build Python wheel (requires maturin)
build-python:
    cd python && maturin build --release

# Run the Python binding tests (requires maturin and pytest)
test-python:
    cd python && maturin develop && pytest -q

# Build WASM package (requires wasm-pack)
build-wasm:
    wasm-pack build crates/marz-wasm --out-dir ../../js/pkg --target web --profile wasm-release

# Build JS wrapper (requires npm install in js/)
build-js:
    cd js && npm run build

# Build all bindings
build-all: build-python build-wasm build-js
