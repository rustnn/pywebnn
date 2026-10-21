//! Internal rustnn `MLContext` state shared by pywebnn bindings.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use rustnn::graph::{pack_int4, pack_uint4, unpack_int4, unpack_uint4, DataType};
use rustnn::mlcontext::{
    Backend, MLContext, MLContextOptions, MLPowerPreference, MLTensor, MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;
use std::collections::BTreeMap;
use std::sync::Once;

use super::graph::PyMLGraph;
use super::operand::parse_data_type;
use super::tensor::RustnnTensor;

/// Runtime state owned by a Python context. Graphs own their compiled resources.
pub(crate) struct ContextState {
    pub ml_context: MLContext<'static>,
}

/// Enable RustNN's existing log output only when the embedding application
/// explicitly configures `RUST_LOG`. A library must not install a global logger
/// during normal imports because Python applications may own that logger.
fn init_rust_logging_if_requested() {
    static LOGGER_INIT: Once = Once::new();
    if std::env::var_os("RUST_LOG").is_some() {
        LOGGER_INIT.call_once(|| {
            let _ = pretty_env_logger::try_init();
        });
    }
}

pub(crate) fn map_rustnn_error(err: rustnn::error::Error) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

pub(crate) fn parse_power_preference(s: &str) -> PyResult<MLPowerPreference> {
    match s {
        "default" => Ok(MLPowerPreference::Default),
        "high-performance" => Ok(MLPowerPreference::HighPerformance),
        "low-power" => Ok(MLPowerPreference::LowPower),
        _ => Err(PyValueError::new_err(format!(
            "Invalid power_preference: {s}. Use 'default', 'high-performance', or 'low-power'"
        ))),
    }
}

pub(crate) fn build_context_options(
    power_preference: &str,
    accelerated: bool,
    backend: &str,
) -> PyResult<MLContextOptions> {
    let options = MLContextOptions::new(parse_power_preference(power_preference)?, accelerated);
    match backend {
        "auto" => Ok(options),
        "onnx" => Ok(options.with_rustnn_backend_hint(Backend::Onnx)),
        "trtx" => Ok(options.with_rustnn_backend_hint(Backend::Trtx)),
        "coreml" => Ok(options.with_rustnn_backend_hint(Backend::Coreml)),
        "litert" => Ok(options.with_rustnn_backend_hint(Backend::Litert)),
        "cann" => Ok(options.with_rustnn_backend_hint(Backend::Cann)),
        _ => Err(PyValueError::new_err("Unknown RustNN backend")),
    }
}

pub(crate) fn data_type_to_ml_operand(dt: DataType) -> PyResult<MLOperandDataType> {
    MLOperandDataType::try_from(dt).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn tensor_element_count(tensor: &MLTensor) -> usize {
    tensor.shape().iter().product::<u64>() as usize
}

fn packed_storage_bytes(data_type: DataType, elements: usize) -> PyResult<usize> {
    data_type
        .storage_byte_length(elements)
        .ok_or_else(|| PyValueError::new_err(format!("invalid element count for {data_type:?}")))
}

fn write_pod_buffer<T>(
    state: &mut ContextState,
    tensor: &MLTensor,
    array: &Bound<'_, PyAny>,
    expected_len: usize,
) -> PyResult<()>
where
    T: Copy + bytemuck::Pod,
{
    let bytes = array.call_method0("tobytes")?;
    let raw = bytes.cast::<PyBytes>()?.as_bytes();
    let typed: &[T] = bytemuck::try_cast_slice(raw).map_err(|_| {
        PyValueError::new_err("write_tensor: invalid buffer size for tensor data type")
    })?;
    if typed.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: expected {expected_len} elements, got {}",
            typed.len()
        )));
    }
    state
        .ml_context
        .write_tensor(tensor, typed)
        .map_err(map_rustnn_error)
}

fn extract_int4_logical_values(flat: &Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
    let n: usize = flat.getattr("size")?.extract()?;
    let bytes = flat.call_method0("tobytes")?;
    let raw = bytes.cast::<PyBytes>()?.as_bytes();
    let slice: &[i8] = bytemuck::try_cast_slice(raw)
        .map_err(|_| PyValueError::new_err("write_tensor: invalid int4 buffer size"))?;
    if slice.len() != n {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: expected {n} int4 elements, got {}",
            slice.len()
        )));
    }
    for (idx, &value) in slice.iter().enumerate() {
        if !(-8..=7).contains(&(value as i32)) {
            return Err(PyValueError::new_err(format!(
                "int4 values must be in [-8, 7]; got {value} at index {idx}"
            )));
        }
    }
    Ok(slice.iter().map(|&v| v as i32).collect())
}

fn extract_uint4_logical_values(flat: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let n: usize = flat.getattr("size")?.extract()?;
    let bytes = flat.call_method0("tobytes")?;
    let raw = bytes.cast::<PyBytes>()?.as_bytes();
    let slice: &[u8] = bytemuck::try_cast_slice(raw)
        .map_err(|_| PyValueError::new_err("write_tensor: invalid uint4 buffer size"))?;
    if slice.len() != n {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: expected {n} uint4 elements, got {}",
            slice.len()
        )));
    }
    for (idx, &value) in slice.iter().enumerate() {
        if value > 15 {
            return Err(PyValueError::new_err(format!(
                "uint4 values must be in [0, 15]; got {value} at index {idx}"
            )));
        }
    }
    Ok(slice.to_vec())
}

/// Pack NumPy logical values into WebNN constant/tensor bytes for 4-bit types.
pub(crate) fn pack_numpy_to_4bit_bytes(
    array: Bound<'_, PyAny>,
    data_type: DataType,
) -> PyResult<Vec<u8>> {
    let flat = array.call_method0("flatten")?;
    match data_type {
        DataType::Int4 => Ok(pack_int4(&extract_int4_logical_values(&flat)?)),
        DataType::Uint4 => Ok(pack_uint4(&extract_uint4_logical_values(&flat)?)),
        _ => Err(PyValueError::new_err(
            "pack_numpy_to_4bit_bytes expects int4 or uint4",
        )),
    }
}

pub(crate) fn ml_tensor_descriptor(
    shape: &[u32],
    data_type: &str,
    readable: bool,
    writable: bool,
) -> PyResult<MLTensorDescriptor> {
    let dt = parse_data_type(data_type)?;
    let ml_dt = data_type_to_ml_operand(dt)?;
    let shape_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
    let mut desc = MLTensorDescriptor::new(ml_dt, shape_u64);
    desc.set_readable(readable);
    desc.set_writable(writable);
    Ok(desc)
}

impl ContextState {
    pub fn new(options: MLContextOptions) -> PyResult<Self> {
        init_rust_logging_if_requested();
        // The safe RustNN constructor owns its runtime; options are not borrowed.
        let ml_context: MLContext<'static> =
            MLContext::create(&options).map_err(map_rustnn_error)?;
        Ok(Self { ml_context })
    }
}

fn coerce_contiguous_write_array<'py>(
    numpy: &Bound<'py, PyModule>,
    data: &Bound<'py, PyAny>,
    dtype_str: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let array = numpy.call_method1("asarray", (data,))?;
    let typed = array.call_method1("astype", (dtype_str,))?;
    numpy.call_method1("ascontiguousarray", (typed,))
}

fn write_numpy_to_ml_tensor(
    _py: Python<'_>,
    state: &mut ContextState,
    tensor: &MLTensor,
    data: Bound<'_, PyAny>,
    numpy: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let dt = tensor.data_type();
    let element_count = tensor_element_count(tensor);

    if matches!(dt, MLOperandDataType::Int4 | MLOperandDataType::Uint4) {
        let data_type = match dt {
            MLOperandDataType::Int4 => DataType::Int4,
            MLOperandDataType::Uint4 => DataType::Uint4,
            _ => unreachable!(),
        };
        let carrier_dtype = if data_type == DataType::Int4 {
            "int8"
        } else {
            "uint8"
        };
        let array = coerce_contiguous_write_array(numpy, &data, carrier_dtype)?;
        let packed = pack_numpy_to_4bit_bytes(array, data_type)?;
        return state
            .ml_context
            .write_tensor(tensor, &packed)
            .map_err(map_rustnn_error);
    }

    let flat = coerce_contiguous_write_array(numpy, &data, dt.as_str())?;

    match dt {
        MLOperandDataType::Float32 => write_pod_buffer::<f32>(state, tensor, &flat, element_count),
        MLOperandDataType::Float16 => {
            let bits = flat.call_method1("view", ("uint16",))?;
            write_pod_buffer::<u16>(state, tensor, &bits, element_count)
        }
        MLOperandDataType::Int32 => write_pod_buffer::<i32>(state, tensor, &flat, element_count),
        MLOperandDataType::Uint32 => write_pod_buffer::<u32>(state, tensor, &flat, element_count),
        MLOperandDataType::Int8 => write_pod_buffer::<i8>(state, tensor, &flat, element_count),
        MLOperandDataType::Uint8 => write_pod_buffer::<u8>(state, tensor, &flat, element_count),
        MLOperandDataType::Int64 => write_pod_buffer::<i64>(state, tensor, &flat, element_count),
        MLOperandDataType::Uint64 => write_pod_buffer::<u64>(state, tensor, &flat, element_count),
        MLOperandDataType::Int4 | MLOperandDataType::Uint4 => unreachable!(),
    }
}

fn numpy_array_from_bytes<'py>(
    py: Python<'py>,
    numpy: &Bound<'py, PyModule>,
    bytes: &[u8],
    dtype: &str,
    shape_tuple: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    let py_bytes = PyBytes::new(py, bytes);
    let array = numpy.call_method1("frombuffer", (py_bytes, dtype))?;
    array.call_method1("reshape", (shape_tuple,))
}

fn read_typed_ml_tensor<'py, T: bytemuck::Pod + Default + Clone>(
    py: Python<'py>,
    state: &mut ContextState,
    tensor: &MLTensor,
    numpy: &Bound<'py, PyModule>,
    dtype: &str,
    shape: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    let count = tensor.shape().iter().product::<u64>() as usize;
    let mut buffer = vec![T::default(); count];
    state
        .ml_context
        .read_tensor(tensor, &mut buffer)
        .map_err(map_rustnn_error)?;
    numpy_array_from_bytes(py, numpy, bytemuck::cast_slice(&buffer), dtype, shape)
}

fn read_ml_tensor_to_numpy<'py>(
    py: Python<'py>,
    state: &mut ContextState,
    tensor: &MLTensor,
    data_type: DataType,
    numpy: &Bound<'py, PyModule>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape: Vec<i64> = tensor.shape().iter().map(|&d| d as i64).collect();
    let shape_tuple = PyTuple::new(py, shape)?;

    match data_type {
        DataType::Float32 => {
            read_typed_ml_tensor::<f32>(py, state, tensor, numpy, "float32", &shape_tuple)
        }
        DataType::Float16 => {
            read_typed_ml_tensor::<u16>(py, state, tensor, numpy, "float16", &shape_tuple)
        }
        DataType::Int32 => {
            read_typed_ml_tensor::<i32>(py, state, tensor, numpy, "int32", &shape_tuple)
        }
        DataType::Uint32 => {
            read_typed_ml_tensor::<u32>(py, state, tensor, numpy, "uint32", &shape_tuple)
        }
        DataType::Int8 => {
            read_typed_ml_tensor::<i8>(py, state, tensor, numpy, "int8", &shape_tuple)
        }
        DataType::Uint8 => {
            read_typed_ml_tensor::<u8>(py, state, tensor, numpy, "uint8", &shape_tuple)
        }
        DataType::Int64 => {
            read_typed_ml_tensor::<i64>(py, state, tensor, numpy, "int64", &shape_tuple)
        }
        DataType::Uint64 => {
            read_typed_ml_tensor::<u64>(py, state, tensor, numpy, "uint64", &shape_tuple)
        }
        DataType::Int4 => {
            let elements = tensor_element_count(tensor);
            let byte_len = packed_storage_bytes(data_type, elements)?;
            let mut packed = vec![0u8; byte_len];
            state
                .ml_context
                .read_tensor(tensor, &mut packed)
                .map_err(map_rustnn_error)?;
            let logical = unpack_int4(&packed, elements);
            let values: Vec<i8> = logical.into_iter().map(|v| v as i8).collect();
            let array = numpy.call_method1("array", (values,))?;
            array
                .call_method1("astype", ("int8",))?
                .call_method1("reshape", (shape_tuple,))
        }
        DataType::Uint4 => {
            let elements = tensor_element_count(tensor);
            let byte_len = packed_storage_bytes(data_type, elements)?;
            let mut packed = vec![0u8; byte_len];
            state
                .ml_context
                .read_tensor(tensor, &mut packed)
                .map_err(map_rustnn_error)?;
            let logical = unpack_uint4(&packed, elements);
            let values: Vec<i32> = logical.into_iter().map(|v| v as i32).collect();
            let array = numpy.call_method1("array", (values,))?;
            array
                .call_method1("astype", ("uint8",))?
                .call_method1("reshape", (shape_tuple,))
        }
    }
}

pub(crate) fn create_rustnn_tensor(
    state: &mut ContextState,
    shape: Vec<u32>,
    data_type: &str,
    readable: bool,
    writable: bool,
) -> PyResult<RustnnTensor> {
    let desc = ml_tensor_descriptor(&shape, data_type, readable, writable)?;
    let tensor = state
        .ml_context
        .create_tensor(&desc)
        .map_err(map_rustnn_error)?;
    Ok(RustnnTensor { tensor })
}

pub(crate) fn read_rustnn_tensor<'py>(
    py: Python<'py>,
    state: &mut ContextState,
    tensor: &RustnnTensor,
) -> PyResult<Bound<'py, PyAny>> {
    let numpy = py.import("numpy")?;
    let data_type = DataType::from(tensor.tensor.data_type());
    read_ml_tensor_to_numpy(py, state, &tensor.tensor, data_type, &numpy)
}

pub(crate) fn write_rustnn_tensor(
    py: Python,
    state: &mut ContextState,
    tensor: &RustnnTensor,
    data: Bound<'_, PyAny>,
) -> PyResult<()> {
    let numpy = py.import("numpy")?;
    write_numpy_to_ml_tensor(py, state, &tensor.tensor, data, &numpy)
}

pub(crate) fn resize_rustnn_tensor(
    state: &mut ContextState,
    inner: &mut RustnnTensor,
    shape: &[u32],
) -> PyResult<()> {
    let shape_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
    state
        .ml_context
        .rustnn_resize_tensor(&mut inner.tensor, &shape_u64)
        .map_err(map_rustnn_error)?;
    Ok(())
}

pub(crate) fn set_rustnn_tensor_capacity(
    state: &mut ContextState,
    inner: &mut RustnnTensor,
    max_shape: &[u32],
) -> PyResult<()> {
    let shape_u64: Vec<u64> = max_shape.iter().map(|&d| d as u64).collect();
    state
        .ml_context
        .rustnn_set_tensor_capacity(&mut inner.tensor, &shape_u64)
        .map_err(map_rustnn_error)?;
    Ok(())
}

/// Dispatch with persistent `MLTensor` bindings (no numpy round-trip).
pub(crate) fn dispatch_with_ml_tensors(
    py: Python,
    state: &mut ContextState,
    graph: &mut PyMLGraph,
    context: &Py<super::context::PyMLContext>,
    inputs: &Bound<'_, PyDict>,
    outputs: &Bound<'_, PyDict>,
) -> PyResult<()> {
    if !graph.context.bind(py).is(context.bind(py)) {
        return Err(PyValueError::new_err(
            "Graph belongs to a different MLContext",
        ));
    }

    let mut input_names = Vec::new();
    let mut input_tensors = Vec::new();
    for (key, value) in inputs.iter() {
        let name: String = key.extract()?;
        let py_tensor = value.cast::<super::tensor::PyMLTensor>()?;
        let bound = py_tensor.borrow();
        input_names.push(name);
        input_tensors.push(bound.inner.tensor.clone());
    }

    let mut output_names = Vec::new();
    let mut output_tensors = Vec::new();
    for (key, value) in outputs.iter() {
        let name: String = key.extract()?;
        let py_tensor = value.cast::<super::tensor::PyMLTensor>()?;
        let bound = py_tensor.borrow();
        output_names.push(name);
        output_tensors.push(bound.inner.tensor.clone());
    }

    let input_refs: BTreeMap<&str, &MLTensor> = input_names
        .iter()
        .zip(&input_tensors)
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    let output_refs: BTreeMap<&str, &MLTensor> = output_names
        .iter()
        .zip(&output_tensors)
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();

    state
        .ml_context
        .dispatch(&mut graph.inner, &input_refs, &output_refs)
        .map_err(map_rustnn_error)
}
