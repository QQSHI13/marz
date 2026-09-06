#!/usr/bin/env bash
# Generate the Snowball stemmers as Rust, and vendor the runtime they need.
#
# Usage: bash scripts/vendor-stemmers.sh [SNOWBALL_REF]
#   SNOWBALL_REF defaults to the pinned ref in SNOWBALL_REF.
#
# Clones snowballstem/snowball at the ref, builds the snowball compiler, and
# generates every algorithm to crates/marz-core/src/stemmers/st_<lang>.rs.
#
# Why generate rather than depend on the `rust-stemmers` crate: that crate
# carries 18 algorithms where upstream has 37, and it would be the first
# non-serde dependency in marz-core. The generated code needs nothing but
# `SnowballEnv` and `Among` — a ~500-line, std-only runtime vendored alongside
# it — so 37 languages cost zero dependencies.
#
# Generated output is COMMITTED. A normal `cargo build` needs no network, no C
# compiler and no Snowball checkout; this script runs only when the pinned ref
# changes. CI re-runs it and asserts the committed output matches byte for byte,
# which is what keeps generated code from drifting away from its source.
set -euo pipefail

cd "$(dirname "$0")/.."

REF="${1:-$(cat SNOWBALL_REF)}"
OUT="crates/marz-core/src/stemmers"
echo "Generating stemmers from snowball ${REF}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone --depth 1 --branch "${REF}" \
  https://github.com/snowballstem/snowball.git "${TMP}/snowball" \
  >/dev/null 2>&1

make -C "${TMP}/snowball" snowball >/dev/null

# Start clean so an algorithm removed upstream does not linger as a stale file
# that still compiles and still gets dispatched to.
rm -rf "${OUT}"
mkdir -p "${OUT}"

# The runtime: `SnowballEnv` (the cursor and the string being stemmed) and
# `Among` (the generated tables' lookup entries). Nothing else, and no
# dependencies beyond `std::borrow::Cow`.
cp -r "${TMP}/snowball/rust/src/snowball" "${OUT}/snowball"

# Upstream's runtime expects to be a crate root reached through a build script:
# `snowball/algorithms/mod.rs` is `include!(concat!(env!("OUT_DIR"), ...))`, which
# fails to compile anywhere else because OUT_DIR is not set. Marz declares its
# stemmer modules directly in mod.rs, so that directory has no job here.
rm -rf "${OUT}/snowball/algorithms"
sed -i '/^pub mod algorithms;/d' "${OUT}/snowball/mod.rs"

# The generator emits 2015-edition paths — a bare `snowball::` meaning "the crate
# named snowball". Inside marz-core these are modules, so they need `crate::` or
# `super::` to resolve at all.
sed -i 's|^pub use snowball::|pub use self::|' "${OUT}/snowball/mod.rs"
sed -i 's|^use snowball::|use super::|' \
  "${OUT}/snowball/among.rs" "${OUT}/snowball/snowball_env.rs"

# Derive the algorithm list from the checked-out tree rather than hardcoding it,
# so the set tracks the pinned ref instead of this script's age.
for sbl in "${TMP}"/snowball/algorithms/*.sbl; do
  lang=$(basename "${sbl}" .sbl)
  "${TMP}/snowball/snowball" "${sbl}" -rust -o "${OUT}/st_${lang}.rs"
done

# Same 2015-edition problem in the generated stemmers.
sed -i 's|^use snowball::|use crate::stemmers::snowball::|' "${OUT}"/st_*.rs

# `#[no_mangle]` is an unsafe attribute as of Rust 2024 and rustc rejects the
# bare form outright. Exported symbols are meaningless here anyway — these are
# internal modules, not a C ABI surface — but rewriting is less invasive than
# stripping the attribute and risks nothing.
sed -i 's|^#\[no_mangle\]|#[unsafe(no_mangle)]|' "${OUT}"/st_*.rs

# mod.rs is generated too: the module list must match the files that exist, and
# maintaining it by hand would be one more thing to drift.
{
  cat <<'EOF'
//! Snowball stemmers, generated from the upstream `.sbl` algorithms.
//!
//! DO NOT EDIT. Regenerate with `scripts/vendor-stemmers.sh`; CI asserts this
//! directory matches a fresh build at the ref in `SNOWBALL_REF`.
//!
//! Each `st_<language>` module exposes `stem(&mut SnowballEnv) -> bool`, which
//! mutates the environment's current string in place. `crate::languages`
//! wraps that in the `Language` trait; nothing outside this crate calls it
//! directly.
//!
//! The generated code is machine output: it trips most style lints and reads
//! nothing like the rest of this codebase. Lints are silenced here rather than
//! in the generator, since the generator is upstream's.

//! `missing_docs` is in the allow list because the crate warns on it globally
//! and the generator does not emit doc comments. Documenting machine output by
//! hand would only guarantee the docs and the code disagree after the next
//! regeneration.

#![allow(
    clippy::all,
    clippy::pedantic,
    non_snake_case,
    missing_docs,
    unused_assignments,
    unused_mut,
    unused_parens,
    unused_variables,
    dead_code
)]

pub mod snowball;

EOF
  for f in "${OUT}"/st_*.rs; do
    echo "pub mod $(basename "${f}" .rs);"
  done
} > "${OUT}/mod.rs"

echo "${REF}" > SNOWBALL_REF
echo "Generated $(ls "${OUT}"/st_*.rs | wc -l) stemmers from ${REF}"
