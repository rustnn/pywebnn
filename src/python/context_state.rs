//! Internal rustnn `MLContext` state shared by pywebnn bindings.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use rustnn::graph::{
    get_static_or_max_size, pack_int4, pack_uint4, unpack_int4, unpack_uint4, DataType, GraphInfo,
};
use rustnn::mlcontext::{
    Backend, MLContext, MLContextOptions, MLGraph, MLGraphBuilder, MLPowerPreference, MLTensor,
    MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::Operation;
use std::collections::{BTreeMap, HashMap};
use std::sync::Once;

use super::graph::PyMLGraph;
use super::operand::parse_data_type;
use super::tensor::RustnnTensor;

/// Owned runtime state: context and compiled graphs share one lifetime region.
pub(crate) struct ContextState {
    pub ml_context: MLContext<'static>,
    /// Keeps graph IR alive for `'static` references held by compiled backends.
    _graph_info_storage: Vec<&'static GraphInfo>,
    pub graphs: Vec<Option<MLGraph<'static>>>,
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

#[allow(dead_code)]
pub(crate) fn power_preference_to_string(pref: MLPowerPreference) -> String {
    match pref {
        MLPowerPreference::Default => "default".to_string(),
        MLPowerPreference::HighPerformance => "high-performance".to_string(),
        MLPowerPreference::LowPower => "low-power".to_string(),
    }
}

/// Map accelerated + legacy `device_type` hints to `MLContextOptions` (rustnn has no device_type field).
pub(crate) fn build_context_options(
    power_preference: &str,
    accelerated_requested: bool,
    device_type: &str,
    backend: &str,
) -> PyResult<MLContextOptions> {
    let power_preference = parse_power_preference(power_preference)?;
    let accelerated = match device_type {
        "cpu" => false,
        "gpu" | "npu" | "auto" => accelerated_requested,
        other => {
            return Err(PyValueError::new_err(format!(
                "Invalid device_type: {other}. Use 'auto', 'cpu', 'gpu', or 'npu' (npu not supported on rustnn MLContext path)"
            )));
        }
    };
    let options = MLContextOptions::new(power_preference, accelerated);
    match backend {
        "auto" if device_type == "cpu" || !accelerated => {
            Ok(options.with_rustnn_backend_hint(Backend::Onnx))
        }
        "auto" => Ok(options),
        "onnx" => Ok(options.with_rustnn_backend_hint(Backend::Onnx)),
        "trtx" => Ok(options.with_rustnn_backend_hint(Backend::Trtx)),
        "coreml" => Ok(options.with_rustnn_backend_hint(Backend::Coreml)),
        "litert" => Ok(options.with_rustnn_backend_hint(Backend::Litert)),
        "cann" => Ok(options.with_rustnn_backend_hint(Backend::Cann)),
        other => Err(PyValueError::new_err(format!(
            "Invalid backend: {other}. Use 'auto', 'onnx', 'trtx', 'coreml', 'litert', or 'cann'"
        ))),
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
        // SAFETY: `MLContext` and backend builders do not expose references tied to caller
        // stack frames; storing in this struct is the intended ownership model for pywebnn.
        let ml_context = MLContext::create(&options).map_err(map_rustnn_error)?;
        let ml_context =
            unsafe { std::mem::transmute::<MLContext<'_>, MLContext<'static>>(ml_context) };
        Ok(Self {
            ml_context,
            _graph_info_storage: Vec::new(),
            graphs: Vec::new(),
        })
    }

    pub fn accelerated(&self) -> bool {
        self.ml_context.accelerated()
    }

    pub fn compile_graph(&mut self, graph_info: GraphInfo) -> PyResult<usize> {
        let info: &'static GraphInfo = Box::leak(Box::new(graph_info));
        self._graph_info_storage.push(info);
        let mut builder = MLGraphBuilder::new(&mut self.ml_context).map_err(map_rustnn_error)?;
        let ml_graph = builder
            .build_graph_info(info.clone())
            .map_err(map_rustnn_error)?;
        let ml_graph = unsafe { std::mem::transmute::<MLGraph<'_>, MLGraph<'static>>(ml_graph) };
        let slot = self.graphs.len();
        self.graphs.push(Some(ml_graph));
        Ok(slot)
    }

    pub fn dispatch_graph(
        &mut self,
        graph_slot: usize,
        inputs: &BTreeMap<&str, &MLTensor>,
        outputs: &BTreeMap<&str, &MLTensor>,
    ) -> PyResult<()> {
        let graph = self
            .graphs
            .get_mut(graph_slot)
            .and_then(|g| g.as_mut())
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!("Invalid compiled graph slot: {graph_slot}"))
            })?;
        self.ml_context
            .dispatch(graph, inputs, outputs)
            .map_err(map_rustnn_error)
    }
}

pub(crate) fn validate_slice_ops(graph: &PyMLGraph) -> PyResult<()> {
    for (idx, op) in graph.graph_info.operations.iter().enumerate() {
        if let Operation::Slice {
            input,
            starts,
            sizes,
            ..
        } = op
        {
            if starts.len() != sizes.len() {
                return Err(PyValueError::new_err(format!(
                    "Slice operation at index {idx} has mismatched starts/sizes (starts.len()={}, sizes.len()={}).",
                    starts.len(),
                    sizes.len()
                )));
            }
            let both_empty = starts.is_empty() && sizes.is_empty();
            if both_empty {
                let input_rank = graph
                    .graph_info
                    .operand(*input)
                    .map(|operand| operand.descriptor.static_or_max_shape().len())
                    .unwrap_or(usize::MAX);
                if input_rank != 0 {
                    return Err(PyValueError::new_err(format!(
                        "Slice operation at index {idx} has empty starts/sizes but input rank is {input_rank} (only 0D no-op slice may have empty starts/sizes)."
                    )));
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn ensure_graph_compiled(
    py: Python,
    state: &mut ContextState,
    graph: &mut PyMLGraph,
    context: &Py<super::context::PyMLContext>,
) -> PyResult<usize> {
    if let Some(slot) = graph.graph_slot {
        return Ok(slot);
    }
    let slot = state.compile_graph(graph.graph_info.clone())?;
    graph.graph_slot = Some(slot);
    graph.context = Some(context.clone_ref(py));
    Ok(slot)
}

fn operand_input_name(graph: &GraphInfo, input_id: u32) -> String {
    graph
        .operands
        .get(input_id as usize)
        .and_then(|op| op.name.clone())
        .unwrap_or_else(|| format!("input_{input_id}"))
}

fn operand_output_name(graph: &GraphInfo, output_id: u32) -> String {
    graph
        .operands
        .get(output_id as usize)
        .and_then(|op| op.name.clone())
        .unwrap_or_else(|| format!("output_{output_id}"))
}

/// Validate that each provided numpy input matches the graph's static input descriptor.
///
/// Validation only: does not mutate `GraphInfo` or run shape inference (handled at load
/// via rustnn `from_graph_json`). Dynamic graphs must use `resize_tensor` + `dispatch`.
fn validate_compute_static_input_shapes(
    graph_info: &GraphInfo,
    named_input_shapes: &HashMap<String, Vec<u32>>,
) -> PyResult<()> {
    for &input_id in &graph_info.input_operands {
        let input_name = graph_info.operands[input_id as usize]
            .name
            .clone()
            .unwrap_or_else(|| operand_input_name(graph_info, input_id));
        let Some(runtime_shape) = named_input_shapes.get(&input_name) else {
            continue;
        };

        let descriptor = &graph_info.operands[input_id as usize].descriptor;

        if descriptor.shape.is_empty() {
            return Err(PyValueError::new_err(format!(
                "Input '{input_name}' has no shape in the graph; cannot run compute()"
            )));
        }

        if descriptor.has_dynamic_dimensions() {
            return Err(PyValueError::new_err(format!(
                "Input '{input_name}' has dynamic dimensions in the graph. \
                 compute() is for static shapes only; use resize_tensor + dispatch for variable sizes."
            )));
        }

        let expected = descriptor.static_shape().ok_or_else(|| {
            PyValueError::new_err(format!(
                "Input '{input_name}' has a non-static graph shape that could not be resolved"
            ))
        })?;
        if runtime_shape != &expected {
            return Err(PyValueError::new_err(format!(
                "Input '{input_name}' shape {runtime_shape:?} does not match graph input shape {expected:?}. \
                 Static graph inputs cannot change at compute(); use resize_tensor + dispatch for variable sizes."
            )));
        }
    }

    Ok(())
}

fn operand_shape_for_compute(graph_info: &GraphInfo, operand_id: u32) -> PyResult<Vec<u32>> {
    let operand = &graph_info.operands[operand_id as usize];
    if operand.descriptor.shape.is_empty() {
        let name = operand
            .name
            .clone()
            .unwrap_or_else(|| format!("operand_{operand_id}"));
        return Err(PyValueError::new_err(format!(
            "No shape available for '{name}'; reload the graph or fix the model export (output shapes are checked at load)"
        )));
    }
    Ok(operand.descriptor.static_or_max_shape())
}

fn numpy_shape_u32(array: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    let shape_attr = array.getattr("shape")?;
    let dims: Vec<usize> = if let Ok(tuple) = shape_attr.extract::<Vec<usize>>() {
        tuple
    } else {
        let len: usize = shape_attr.len()?;
        (0..len)
            .map(|i| shape_attr.get_item(i)?.extract::<usize>())
            .collect::<PyResult<_>>()?
    };
    Ok(dims.into_iter().map(|d| d as u32).collect())
}

fn numpy_dtype_str(dt: DataType) -> PyResult<&'static str> {
    Ok(data_type_to_ml_operand(dt)?.as_str())
}

/// One-shot `compute()` for **static** graphs: validates numpy inputs against load-time
/// descriptors, then dispatches with ephemeral tensors. Variable sizes → `resize_tensor` + `dispatch`.
pub(crate) fn compute_with_dispatch(
    py: Python,
    state: &mut ContextState,
    graph: &mut PyMLGraph,
    context: &Py<super::context::PyMLContext>,
    inputs: &Bound<'_, PyDict>,
) -> PyResult<Py<PyDict>> {
    validate_slice_ops(graph)?;

    let numpy = py.import("numpy")?;

    let mut named_input_shapes: HashMap<String, Vec<u32>> = HashMap::new();
    for &input_id in &graph.graph_info.input_operands {
        let input_name = graph.graph_info.operands[input_id as usize]
            .name
            .clone()
            .unwrap_or_else(|| operand_input_name(&graph.graph_info, input_id));
        let has_empty_dimension = graph.graph_info.operands[input_id as usize]
            .descriptor
            .shape
            .iter()
            .any(|d| get_static_or_max_size(d) == 0);
        let is_kv_input = input_name.starts_with("past_key_values_");
        if has_empty_dimension && is_kv_input {
            continue;
        }
        if let Some(array) = inputs.get_item(&input_name)? {
            named_input_shapes.insert(input_name, numpy_shape_u32(&array)?);
        }
    }

    validate_compute_static_input_shapes(&graph.graph_info, &named_input_shapes)?;

    let graph_slot = ensure_graph_compiled(py, state, graph, context)?;
    let graph_info = &graph.graph_info;

    let mut input_tensors: Vec<(String, MLTensor)> = Vec::new();

    for &input_id in &graph_info.input_operands {
        let input_op = graph_info.operands.get(input_id as usize).ok_or_else(|| {
            PyValueError::new_err(format!("Input operand {input_id} not found in graph"))
        })?;

        let input_name = input_op
            .name
            .clone()
            .unwrap_or_else(|| operand_input_name(graph_info, input_id));

        let has_empty_dimension = input_op
            .descriptor
            .shape
            .iter()
            .any(|d| get_static_or_max_size(d) == 0);
        let is_kv_input = input_name.starts_with("past_key_values_");
        if has_empty_dimension && is_kv_input {
            continue;
        }

        let array = inputs
            .get_item(&input_name)?
            .ok_or_else(|| PyValueError::new_err(format!("Missing input: {input_name}")))?;

        let shape = numpy_shape_u32(&array)?;
        let dtype_str = numpy_dtype_str(input_op.descriptor.data_type)?;
        let desc = ml_tensor_descriptor(&shape, dtype_str, true, true)?;
        let tensor = state
            .ml_context
            .create_tensor(&desc)
            .map_err(map_rustnn_error)?;

        write_numpy_to_ml_tensor(py, state, &tensor, &desc, array, &numpy)?;
        input_tensors.push((input_name, tensor));
    }

    let input_refs: BTreeMap<&str, &MLTensor> = input_tensors
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();

    let mut output_tensors: Vec<(String, MLTensor)> = Vec::new();

    for &output_id in &graph_info.output_operands {
        let output_op = graph_info.operands.get(output_id as usize).ok_or_else(|| {
            PyValueError::new_err(format!("Output operand {output_id} not found in graph"))
        })?;
        let output_name = output_op
            .name
            .clone()
            .unwrap_or_else(|| operand_output_name(graph_info, output_id));

        let shape = operand_shape_for_compute(graph_info, output_id)?;
        let dtype_str = numpy_dtype_str(output_op.descriptor.data_type)?;
        let desc = ml_tensor_descriptor(&shape, dtype_str, true, true)?;
        let tensor = state
            .ml_context
            .create_tensor(&desc)
            .map_err(map_rustnn_error)?;
        output_tensors.push((output_name, tensor));
    }

    let output_refs: BTreeMap<&str, &MLTensor> = output_tensors
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();

    state.dispatch_graph(graph_slot, &input_refs, &output_refs)?;

    let result = PyDict::new(py);
    for (name, tensor) in &output_tensors {
        let output_op = graph_info
            .output_operands
            .iter()
            .find_map(|&id| {
                let op = &graph_info.operands[id as usize];
                let n = op
                    .name
                    .clone()
                    .unwrap_or_else(|| operand_output_name(graph_info, id));
                if n == *name {
                    Some(op)
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!("Output metadata missing for {name}"))
            })?;

        let arr =
            read_ml_tensor_to_numpy(py, state, tensor, output_op.descriptor.data_type, &numpy)?;
        result.set_item(name, arr)?;
    }

    Ok(result.into())
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
    desc: &MLTensorDescriptor,
    data: Bound<'_, PyAny>,
    numpy: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let dt = desc.data_type();
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
            let n = tensor.shape().iter().product::<u64>() as usize;
            let mut buf = vec![0f32; n];
            state
                .ml_context
                .read_tensor(tensor, &mut buf)
                .map_err(map_rustnn_error)?;
            let array = numpy.call_method1("array", (buf,))?;
            array.call_method1("reshape", (shape_tuple,))
        }
        DataType::Float16 => {
            let n = tensor.shape().iter().product::<u64>() as usize;
            let mut buf = vec![0u16; n];
            state
                .ml_context
                .read_tensor(tensor, &mut buf)
                .map_err(map_rustnn_error)?;
            let f32s: Vec<f32> = buf
                .iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect();
            let array = numpy.call_method1("array", (f32s,))?;
            array
                .call_method1("astype", ("float16",))?
                .call_method1("reshape", (shape_tuple,))
        }
        DataType::Int32 => {
            let n = tensor.shape().iter().product::<u64>() as usize;
            let mut buf = vec![0i32; n];
            state
                .ml_context
                .read_tensor(tensor, &mut buf)
                .map_err(map_rustnn_error)?;
            let array = numpy.call_method1("array", (buf,))?;
            array.call_method1("reshape", (shape_tuple,))
        }
        DataType::Uint32 => {
            let n = tensor.shape().iter().product::<u64>() as usize;
            let mut buf = vec![0u32; n];
            state
                .ml_context
                .read_tensor(tensor, &mut buf)
                .map_err(map_rustnn_error)?;
            let array = numpy.call_method1("array", (buf,))?;
            array.call_method1("reshape", (shape_tuple,))
        }
        DataType::Int8 => {
            let n = tensor.shape().iter().product::<u64>() as usize;
            let mut buf = vec![0i8; n];
            state
                .ml_context
                .read_tensor(tensor, &mut buf)
                .map_err(map_rustnn_error)?;
            let array = numpy.call_method1("array", (buf,))?;
            array.call_method1("reshape", (shape_tuple,))
        }
        DataType::Uint8 => {
            let n = tensor.shape().iter().product::<u64>() as usize;
            let mut buf = vec![0u8; n];
            state
                .ml_context
                .read_tensor(tensor, &mut buf)
                .map_err(map_rustnn_error)?;
            numpy_array_from_bytes(py, numpy, &buf, "uint8", &shape_tuple)
        }
        DataType::Int64 => {
            let n = tensor.shape().iter().product::<u64>() as usize;
            let mut buf = vec![0i64; n];
            state
                .ml_context
                .read_tensor(tensor, &mut buf)
                .map_err(map_rustnn_error)?;
            let array = numpy.call_method1("array", (buf,))?;
            array.call_method1("reshape", (shape_tuple,))
        }
        DataType::Uint64 => {
            let n = tensor.shape().iter().product::<u64>() as usize;
            let mut buf = vec![0u64; n];
            state
                .ml_context
                .read_tensor(tensor, &mut buf)
                .map_err(map_rustnn_error)?;
            let array = numpy.call_method1("array", (buf,))?;
            array.call_method1("reshape", (shape_tuple,))
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
    Ok(RustnnTensor { tensor, desc })
}

pub(crate) fn read_rustnn_tensor<'py>(
    py: Python<'py>,
    state: &mut ContextState,
    tensor: &RustnnTensor,
) -> PyResult<Bound<'py, PyAny>> {
    let numpy = py.import("numpy")?;
    let data_type = DataType::from(tensor.desc.data_type());
    read_ml_tensor_to_numpy(py, state, &tensor.tensor, data_type, &numpy)
}

pub(crate) fn write_rustnn_tensor(
    py: Python,
    state: &mut ContextState,
    tensor: &RustnnTensor,
    data: Bound<'_, PyAny>,
) -> PyResult<()> {
    let numpy = py.import("numpy")?;
    write_numpy_to_ml_tensor(py, state, &tensor.tensor, &tensor.desc, data, &numpy)
}

fn sync_rustnn_tensor_desc(inner: &mut RustnnTensor) {
    inner.desc.set_shape(inner.tensor.shape().to_vec());
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
    sync_rustnn_tensor_desc(inner);
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
    sync_rustnn_tensor_desc(inner);
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
    validate_slice_ops(graph)?;
    let slot = ensure_graph_compiled(py, state, graph, context)?;

    let mut input_names = Vec::new();
    let mut input_tensors = Vec::new();
    for (key, value) in inputs.iter() {
        let name: String = key.extract()?;
        let py_tensor = value.cast::<super::tensor::PyMLTensor>()?;
        let bound = py_tensor.borrow();
        bound.check_destroyed()?;
        input_names.push(name);
        input_tensors.push(bound.inner.tensor.clone());
    }

    let mut output_names = Vec::new();
    let mut output_tensors = Vec::new();
    for (key, value) in outputs.iter() {
        let name: String = key.extract()?;
        let py_tensor = value.cast::<super::tensor::PyMLTensor>()?;
        let bound = py_tensor.borrow();
        bound.check_destroyed()?;
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

    state.dispatch_graph(slot, &input_refs, &output_refs)
}
