//! Built-in language implementations.

pub mod chinese;
pub mod cjk;
pub mod english;
pub mod japanese;
pub mod korean;

pub use chinese::Chinese;
pub use english::English;
pub use japanese::Japanese;
pub use korean::Korean;
