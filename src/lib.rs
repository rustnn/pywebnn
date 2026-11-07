// Python bindings for rustnn - W3C WebNN implementation
//
// This crate provides PyO3 bindings to expose rustnn functionality to Python.
// All core logic is implemented in the rustnn crate; this is a thin wrapper layer.

use pyo3::prelude::*;

mod python;

#[cfg(feature = "onnx-runtime")]
mod ort_environment;

use python::{PyML, PyMLContext, PyMLGraph, PyMLGraphBuilder, PyMLOperand, PyMLTensor};

/// WebNN Python module
#[pymodule]
fn _rustnn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "onnx-runtime")]
    {
        m.add_function(wrap_pyfunction!(ort_environment::init_global_ort_env, m)?)?;
        m.add(
            "OrtAlreadyInitialized",
            m.py().get_type::<ort_environment::OrtAlreadyInitialized>(),
        )?;
    }
    m.add_class::<PyML>()?;
    m.add_class::<PyMLContext>()?;
    m.add_class::<PyMLGraphBuilder>()?;
    m.add_class::<PyMLOperand>()?;
    m.add_class::<PyMLGraph>()?;
    m.add_class::<PyMLTensor>()?;
    Ok(())
}
