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
        "bfloat16" => Ok(DataType::Bfloat16),
        "float8e4m3" | "float8_e4m3fn" => Ok(DataType::Float8e4m3),
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

/// Resolve native dtype names to NumPy types, including ml_dtypes extensions.
pub(crate) fn numpy_dtype<'py>(py: Python<'py>, dtype: &str) -> PyResult<Bound<'py, PyAny>> {
    match dtype {
        "bfloat16" => py.import("ml_dtypes")?.getattr("bfloat16"),
        "float8e4m3" | "float8_e4m3fn" => py.import("ml_dtypes")?.getattr("float8_e4m3fn"),
        _ => py.import("numpy")?.call_method1("dtype", (dtype,)),
    }
}
