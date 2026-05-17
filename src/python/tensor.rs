//! MLTensor implementation following WebNN MLTensor Explainer

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use rustnn::mlcontext::MLTensorDescriptor;
use std::sync::{Arc, Mutex};

use super::context::PyMLContext;
/// Host-side view of a rustnn `MLTensor` owned by a context.
pub(crate) struct RustnnTensor {
    pub tensor: rustnn::mlcontext::MLTensor,
    pub desc: MLTensorDescriptor,
}

/// MLTensor - opaque typed tensor backed by rustnn runtime storage.
#[pyclass(name = "MLTensor")]
pub struct PyMLTensor {
    pub(crate) context: Py<PyMLContext>,
    pub(crate) inner: RustnnTensor,
    destroyed: Arc<Mutex<bool>>,
}

impl PyMLTensor {
    pub(crate) fn from_rustnn(context: Py<PyMLContext>, inner: RustnnTensor) -> Self {
        Self {
            context,
            inner,
            destroyed: Arc::new(Mutex::new(false)),
        }
    }

    pub(crate) fn check_destroyed(&self) -> PyResult<()> {
        if *self.destroyed.lock().unwrap() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Tensor has been destroyed",
            ));
        }
        Ok(())
    }
}

#[pymethods]
impl PyMLTensor {
    #[getter]
    fn data_type(&self) -> String {
        match self.inner.desc.data_type() {
            rustnn::operator_enums::MLOperandDataType::Float32 => "float32".to_string(),
            rustnn::operator_enums::MLOperandDataType::Float16 => "float16".to_string(),
            rustnn::operator_enums::MLOperandDataType::Int32 => "int32".to_string(),
            rustnn::operator_enums::MLOperandDataType::Uint32 => "uint32".to_string(),
            rustnn::operator_enums::MLOperandDataType::Int8 => "int8".to_string(),
            rustnn::operator_enums::MLOperandDataType::Uint8 => "uint8".to_string(),
            rustnn::operator_enums::MLOperandDataType::Int64 => "int64".to_string(),
            rustnn::operator_enums::MLOperandDataType::Uint64 => "uint64".to_string(),
        }
    }

    #[getter]
    fn shape(&self) -> Vec<u32> {
        self.inner
            .tensor
            .shape()
            .iter()
            .map(|&d| d as u32)
            .collect()
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner
            .tensor
            .shape()
            .iter()
            .product::<u64>() as usize
    }

    #[getter]
    pub(crate) fn readable(&self) -> bool {
        self.inner.desc.readable()
    }

    #[getter]
    pub(crate) fn writable(&self) -> bool {
        self.inner.desc.writable()
    }

    #[getter]
    fn exportable_to_gpu(&self) -> bool {
        false
    }

    fn destroy(&self) -> PyResult<()> {
        let mut destroyed = self.destroyed.lock().unwrap();
        if *destroyed {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Tensor already destroyed",
            ));
        }
        *destroyed = true;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "MLTensor(shape={:?}, dtype={}, readable={}, writable={})",
            self.shape(),
            self.data_type(),
            self.readable(),
            self.writable()
        )
    }
}

/// Device-resident tensor — not supported on the rustnn MLContext execution path.
#[pyclass(name = "MLDeviceTensor")]
pub struct PyMLDeviceTensor;

#[pymethods]
impl PyMLDeviceTensor {
    fn __repr__(&self) -> String {
        "MLDeviceTensor(not supported)".to_string()
    }
}
