//! Direct bindings to RustNN's context and persistent tensor API.
use super::context_state::{
    build_context_options, create_rustnn_tensor, dispatch_with_ml_tensors, map_rustnn_error,
    read_rustnn_tensor, resize_rustnn_tensor, set_rustnn_tensor_capacity, write_rustnn_tensor,
    ContextState,
};
use super::{PyMLGraph, PyMLGraphBuilder, PyMLTensor};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::{Mutex, MutexGuard};

#[pyclass(name = "ML")]
pub struct PyML;

#[pymethods]
impl PyML {
    #[new]
    fn new() -> Self {
        Self
    }

    #[pyo3(signature = (power_preference="default", accelerated=true, backend="auto"))]
    fn create_context(
        &self,
        power_preference: &str,
        accelerated: bool,
        backend: &str,
    ) -> PyResult<PyMLContext> {
        let options = build_context_options(power_preference, accelerated, backend)?;
        Ok(PyMLContext {
            state: Mutex::new(ContextState::new(options)?),
        })
    }
}

#[pyclass(name = "MLContext", unsendable)]
pub struct PyMLContext {
    state: Mutex<ContextState>,
}

#[pymethods]
impl PyMLContext {
    fn create_graph_builder(this: Py<Self>, py: Python<'_>) -> PyResult<PyMLGraphBuilder> {
        PyMLGraphBuilder::new_for_context(py, this)
    }

    /// Load RustNN's graph-file format and compile it for this context.
    fn load_graph(this: Py<Self>, py: Python<'_>, path: &str) -> PyResult<PyMLGraph> {
        let graph = rustnn::load_graph_from_path(path)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        let inner = this
            .bind(py)
            .borrow()
            .runtime_state()?
            .ml_context
            .rustnn_build_graph(graph)
            .map_err(map_rustnn_error)?;
        Ok(PyMLGraph::new(inner, this))
    }

    /// Dispatch using tensors allocated explicitly by the caller (#25).
    fn dispatch(
        this: Py<Self>,
        py: Python<'_>,
        graph: &mut PyMLGraph,
        inputs: &Bound<'_, PyDict>,
        outputs: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let ctx = this.bind(py).borrow();
        let mut state = ctx.runtime_state()?;
        dispatch_with_ml_tensors(py, &mut state, graph, &this, inputs, outputs)
    }

    #[pyo3(signature = (shape, data_type, readable=false, writable=false))]
    fn create_tensor(
        this: Py<Self>,
        py: Python<'_>,
        shape: Vec<u32>,
        data_type: &str,
        readable: bool,
        writable: bool,
    ) -> PyResult<PyMLTensor> {
        let ctx = this.bind(py).borrow();
        let mut state = ctx.runtime_state()?;
        let inner = create_rustnn_tensor(&mut state, shape, data_type, readable, writable)?;
        Ok(PyMLTensor::from_rustnn(this.clone_ref(py), inner))
    }

    fn read_tensor<'py>(
        &self,
        py: Python<'py>,
        tensor: &PyMLTensor,
    ) -> PyResult<Bound<'py, PyAny>> {
        // TODO(rustnn#320): rely on RustNN for tensor/context ownership validation.
        read_rustnn_tensor(py, &mut *self.runtime_state()?, &tensor.inner)
    }

    fn write_tensor(
        &self,
        py: Python<'_>,
        tensor: &PyMLTensor,
        data: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        // TODO(rustnn#320): rely on RustNN for tensor/context ownership validation.
        // TODO: Convert Python/NumPy input before acquiring the runtime mutex.
        // Conversion can invoke user-defined __array__ methods. If one calls back
        // into this context (e.g. read_tensor), it tries to acquire the same
        // non-reentrant mutex and deadlocks while this write still holds it.
        write_rustnn_tensor(py, &mut *self.runtime_state()?, &tensor.inner, data)
    }

    fn resize_tensor(&self, tensor: &mut PyMLTensor, shape: Vec<u32>) -> PyResult<()> {
        resize_rustnn_tensor(&mut *self.runtime_state()?, &mut tensor.inner, &shape)
    }

    fn set_tensor_capacity(&self, tensor: &mut PyMLTensor, max_shape: Vec<u32>) -> PyResult<()> {
        set_rustnn_tensor_capacity(&mut *self.runtime_state()?, &mut tensor.inner, &max_shape)
    }

    #[getter]
    fn accelerated(&self) -> PyResult<bool> {
        Ok(self.runtime_state()?.ml_context.accelerated())
    }

    fn op_support_limits(&self) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "RustNN does not implement op_support_limits yet",
        ))
    }
}

impl PyMLContext {
    pub(crate) fn runtime_state(&self) -> PyResult<MutexGuard<'_, ContextState>> {
        self.state
            .lock()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("MLContext state lock poisoned"))
    }

    pub(crate) fn create_runtime_builder(
        &self,
    ) -> PyResult<rustnn::mlcontext::MLGraphBuilder<'static, 'static>> {
        rustnn::mlcontext::MLGraphBuilder::new(&mut self.runtime_state()?.ml_context)
            .map_err(map_rustnn_error)
    }
}
