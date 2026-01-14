// Python bindings for rustnn - W3C WebNN implementation
//
// This crate provides PyO3 bindings to expose rustnn functionality to Python.
// All core logic is implemented in the rustnn crate; this is a thin wrapper layer.

use pyo3::prelude::*;

mod python;

use python::{
    PyML, PyMLContext, PyMLDeviceTensor, PyMLGraph, PyMLGraphBuilder, PyMLOperand, PyMLTensor,
};

/// WebNN Python module
#[pymodule]
fn _rustnn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyML>()?;
    m.add_class::<PyMLContext>()?;
    m.add_class::<PyMLGraphBuilder>()?;
    m.add_class::<PyMLOperand>()?;
    m.add_class::<PyMLGraph>()?;
    m.add_class::<PyMLTensor>()?;
    m.add_class::<PyMLDeviceTensor>()?;
    Ok(())
}
