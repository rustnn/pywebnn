//! Context-owned native tensor handles.
use super::context::PyMLContext;
use pyo3::prelude::*;

pub(crate) struct RustnnTensor {
    pub tensor: rustnn::mlcontext::MLTensor,
}

#[pyclass(name = "MLTensor")]
pub struct PyMLTensor {
    pub(crate) inner: RustnnTensor,
    pub(crate) _context: Py<PyMLContext>,
}

impl PyMLTensor {
    pub(crate) fn from_rustnn(context: Py<PyMLContext>, inner: RustnnTensor) -> Self {
        Self {
            inner,
            _context: context,
        }
    }
}

#[pymethods]
impl PyMLTensor {
    #[getter]
    fn data_type(&self) -> String {
        self.inner.tensor.data_type().as_str().to_string()
    }
    #[getter]
    fn shape(&self) -> Vec<u64> {
        self.inner.tensor.shape().to_vec()
    }
    #[getter]
    fn readable(&self) -> bool {
        self.inner.tensor.readable()
    }
    #[getter]
    fn writable(&self) -> bool {
        self.inner.tensor.writable()
    }
    /// RustNN has not implemented tensor destruction yet.
    fn destroy(&self) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "RustNN does not implement tensor.destroy yet",
        ))
    }
}
