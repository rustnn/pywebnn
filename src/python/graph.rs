//! Native compiled graph handle and its public I/O descriptors.
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustnn::graph::{Dimension, OperandDescriptor};
use std::collections::HashMap;

#[pyclass(name = "MLGraph", unsendable)]
pub struct PyMLGraph {
    // Drop the native graph before releasing its runtime owner.
    pub(crate) inner: rustnn::mlcontext::MLGraph<'static>,
    pub(crate) context: Py<super::context::PyMLContext>,
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
        value.set_item(
            "data_type",
            super::context_state::data_type_to_ml_operand(descriptor.data_type)?.as_str(),
        )?;
        result.set_item(name, value)?;
    }
    Ok(result)
}

#[pymethods]
impl PyMLGraph {
    #[getter]
    fn input_descriptors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        descriptors(py, &self.inner.input_descriptors)
    }

    #[getter]
    fn output_descriptors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        descriptors(py, &self.inner.output_descriptors)
    }
}

impl PyMLGraph {
    pub(crate) fn new(
        inner: rustnn::mlcontext::MLGraph<'static>,
        context: Py<super::context::PyMLContext>,
    ) -> Self {
        Self { inner, context }
    }
}
