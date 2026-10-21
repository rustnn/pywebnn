//! Opaque native operand handles. RustNN owns operand metadata.
use pyo3::prelude::*;
use rustnn::graph::DataType;

#[pyclass(name = "MLOperand", from_py_object)]
#[derive(Clone)]
pub struct PyMLOperand {
    pub(crate) handle: rustnn::mlcontext::MLOperand,
    pub(crate) builder_id: u64,
}

/// Parse data type string to DataType enum
pub fn parse_data_type(dtype: &str) -> PyResult<DataType> {
    match dtype {
        "int4" => Ok(DataType::Int4),
        "uint4" => Ok(DataType::Uint4),
        "float32" => Ok(DataType::Float32),
        "float16" => Ok(DataType::Float16),
        "int32" => Ok(DataType::Int32),
        "uint32" => Ok(DataType::Uint32),
        "int8" => Ok(DataType::Int8),
        "uint8" => Ok(DataType::Uint8),
        "int64" => Ok(DataType::Int64),
        "uint64" => Ok(DataType::Uint64),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported data type: {}",
            dtype
        ))),
    }
}
