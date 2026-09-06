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

pub mod st_arabic;
pub mod st_armenian;
pub mod st_basque;
pub mod st_catalan;
pub mod st_czech;
pub mod st_danish;
pub mod st_dutch_porter;
pub mod st_dutch;
pub mod st_english;
pub mod st_esperanto;
pub mod st_estonian;
pub mod st_finnish;
pub mod st_french;
pub mod st_german;
pub mod st_greek;
pub mod st_hindi;
pub mod st_hungarian;
pub mod st_indonesian;
pub mod st_irish;
pub mod st_italian;
pub mod st_lithuanian;
pub mod st_lovins;
pub mod st_nepali;
pub mod st_norwegian;
pub mod st_persian;
pub mod st_polish;
pub mod st_porter;
pub mod st_portuguese;
pub mod st_romanian;
pub mod st_russian;
pub mod st_serbian;
pub mod st_sesotho;
pub mod st_spanish;
pub mod st_swedish;
pub mod st_tamil;
pub mod st_turkish;
pub mod st_yiddish;
