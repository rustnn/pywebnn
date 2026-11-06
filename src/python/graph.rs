//! Native compiled graph handle and its public I/O descriptors.
use super::context_state::{map_rustnn_error, ContextState};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustnn::graph::{Dimension, OperandDescriptor};
use std::collections::HashMap;
use std::path::PathBuf;

#[pyclass(name = "MLGraph", unsendable)]
pub struct PyMLGraph {
    // Drop the native graph before releasing its runtime owner.
    pub(crate) inner: Option<rustnn::mlcontext::MLGraph<'static>>,
    pub(crate) context: Option<Py<super::context::PyMLContext>>,
    info: Option<rustnn::GraphInfo>,
}

// Adapt the native graph's public I/O fields without retaining a second copy.
fn descriptors<'py>(
    py: Python<'py>,
    native: &HashMap<String, OperandDescriptor>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    for (name, descriptor) in native {
        let value = PyDict::new(py);
        let shape = PyList::empty(py);
        for dimension in &descriptor.shape {
            match dimension {
                Dimension::Static(size) => shape.append(size)?,
                Dimension::Dynamic(dimension) => {
                    let dynamic = PyDict::new(py);
                    dynamic.set_item("name", &dimension.name)?;
                    dynamic.set_item("maxSize", dimension.max_size)?;
                    shape.append(dynamic)?;
                }
            }
        }
        value.set_item("shape", shape)?;
        value.set_item("data_type", descriptor.data_type.as_str())?;
        result.set_item(name, value)?;
    }
    Ok(result)
}

#[pymethods]
impl PyMLGraph {
    /// Load a graph without initializing a runtime; compile at first dispatch.
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        Ok(Self::uncompiled(
            rustnn::load_graph_from_path(path).map_err(serialization_error)?,
        ))
    }

    #[getter]
    fn input_descriptors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if let Some(inner) = &self.inner {
            return descriptors(py, &inner.input_descriptors);
        }
        let (inputs, _) = self
            .info
            .as_ref()
            .expect("graph info")
            .io_binding_maps()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        descriptors(py, &inputs)
    }

    #[getter]
    fn output_descriptors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if let Some(inner) = &self.inner {
            return descriptors(py, &inner.output_descriptors);
        }
        let (_, outputs) = self
            .info
            .as_ref()
            .expect("graph info")
            .io_binding_maps()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        descriptors(py, &outputs)
    }
}

impl PyMLGraph {
    pub(crate) fn new(
        inner: rustnn::mlcontext::MLGraph<'static>,
        context: Py<super::context::PyMLContext>,
    ) -> Self {
        Self {
            inner: Some(inner),
            context: Some(context),
            info: None,
        }
    }

    pub(crate) fn uncompiled(info: rustnn::GraphInfo) -> Self {
        Self {
            inner: None,
            context: None,
            info: Some(info),
        }
    }

    pub(crate) fn ensure_compiled(
        &mut self,
        py: Python<'_>,
        state: &mut ContextState,
        context: &Py<super::context::PyMLContext>,
    ) -> PyResult<()> {
        if let Some(owner) = &self.context {
            if !owner.bind(py).is(context.bind(py)) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Graph belongs to a different MLContext",
                ));
            }
        }
        if self.inner.is_none() {
            let info = self.info.as_ref().expect("uncompiled graph info");
            self.inner = Some(
                state
                    .ml_context
                    .rustnn_build_graph(info.clone())
                    .map_err(map_rustnn_error)?,
            );
            self.context = Some(context.clone_ref(py));
            self.info = None;
        }
        Ok(())
    }
}

fn serialization_error(error: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
}
