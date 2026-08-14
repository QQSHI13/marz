//! WebAssembly bindings for Marz.

use wasm_bindgen::prelude::*;

/// Placeholder for the WASM searcher.
#[wasm_bindgen]
pub struct Searcher;

#[wasm_bindgen]
impl Searcher {
    /// Create a new, empty searcher.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Searcher
    }

    /// Return a placeholder version string.
    #[wasm_bindgen(getter)]
    pub fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}

impl Default for Searcher {
    fn default() -> Self {
        Self::new()
    }
}
