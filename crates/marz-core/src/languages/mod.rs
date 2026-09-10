//! Built-in language implementations.

pub mod chinese;
pub mod cjk;
pub mod english;
pub mod generic;
pub mod japanese;
pub mod korean;
pub mod nonspacing;
pub mod porter;
pub mod registry;
pub mod snowball;

pub use chinese::Chinese;
pub use english::English;
pub use generic::Generic;
pub use japanese::Japanese;
pub use korean::Korean;
pub use nonspacing::{NonSpacing, NON_SPACING_LANGUAGES};
pub use registry::{canonical_parts, codes, is_supported, resolve, resolve_multi, Resolved};
pub use snowball::{SnowballLanguage, SNOWBALL_LANGUAGES};
