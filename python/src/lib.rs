//! Python bindings for Marz via PyO3/Maturin.

use pyo3::prelude::*;

/// Placeholder builder class exposed to Python.
#[pyclass]
pub struct IndexBuilder;

#[pymethods]
impl IndexBuilder {
    #[new]
    fn new() -> Self {
        IndexBuilder
    }

    fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}

/// Marz Python module.
#[pymodule]
fn _marz(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IndexBuilder>()?;
    Ok(())
}
