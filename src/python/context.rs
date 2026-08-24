//! ML context and backend selection for WebNN API
//!
//! PyO3 macros generate unsafe code that triggers unsafe_op_in_unsafe_fn warnings.
//! This is expected behavior from the macro-generated code.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::useless_conversion)]

use super::context_state::{
    build_context_options, compute_with_dispatch, create_rustnn_tensor, dispatch_with_ml_tensors,
    read_rustnn_tensor, resize_rustnn_tensor, set_rustnn_tensor_capacity, write_rustnn_tensor,
    ContextState,
};
use super::graph::PyMLGraph;
use super::graph_builder::PyMLGraphBuilder;
use super::tensor::{PyMLDeviceTensor, PyMLTensor};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustnn::converters::GraphConverter;
use rustnn::graph::GraphInfo;
use std::sync::Mutex;

/// ML namespace - entry point for WebNN API
#[pyclass(name = "ML")]
pub struct PyML;

#[pymethods]
impl PyML {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Create a new ML context
    ///
    /// Args:
    ///     power_preference: Power preference hint ("default", "high-performance", or "low-power")
    ///     accelerated: Whether to use GPU/NPU acceleration (default: true)
    ///     device_type: Device preference ("auto", "cpu", "gpu", "npu") (default: "auto")
    ///     backend: RustNN backend ("auto", "onnx", "trtx", "coreml", "litert", or "cann")
    ///
    /// Returns:
    ///     MLContext: A new context for graph operations
    ///
    /// Note:
    ///     The accelerated parameter is a hint, not a guarantee. The platform
    ///     decides the actual device allocation based on runtime conditions.
    ///     Query context.accelerated after creation to check if acceleration is available.
    ///     device_type="auto" uses automatic backend selection based on availability.
    ///     device_type="cpu" requests CPU execution.
    ///     device_type="gpu" requests GPU-accelerated execution.
    ///     device_type="npu" requests NPU execution (platform-dependent, e.g. Apple Neural Engine).
    #[pyo3(signature = (power_preference="default", accelerated=true, device_type="auto", backend="auto"))]
    fn create_context(
        &self,
        power_preference: &str,
        accelerated: bool,
        device_type: &str,
        backend: &str,
    ) -> PyResult<PyMLContext> {
        PyMLContext::new(
            power_preference.to_string(),
            accelerated,
            device_type.to_string(),
            backend.to_string(),
        )
    }
}

/// MLContext manages the execution environment for neural network graphs
#[pyclass(name = "MLContext", unsendable)]
pub struct PyMLContext {
    power_preference: String,
    #[allow(dead_code)]
    accelerated_requested: bool,
    device_type: String,
    backend: String,
    state: Mutex<ContextState>,
}

#[pymethods]
impl PyMLContext {
    /// Create a graph builder for constructing computational graphs
    ///
    /// Returns:
    ///     MLGraphBuilder: A new graph builder
    fn create_graph_builder(this: Py<Self>) -> PyResult<PyMLGraphBuilder> {
        Ok(PyMLGraphBuilder::new_for_context(this))
    }

    /// Compute the graph with given inputs using the backend selected at context creation
    ///
    /// This is a synchronous execution method that returns computed results.
    ///
    /// Args:
    ///     graph: The compiled MLGraph to execute
    ///     inputs: Dictionary mapping input names to numpy arrays
    ///     outputs: Dictionary mapping output names to numpy arrays (pre-allocated)
    ///
    /// Returns:
    ///     Dictionary mapping output names to result numpy arrays
    #[pyo3(signature = (graph, inputs, _outputs=None))]
    fn compute(
        this: Py<Self>,
        py: Python,
        graph: &mut PyMLGraph,
        inputs: &Bound<'_, PyDict>,
        _outputs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDict>> {
        let bound = this.bind(py);
        let ctx = bound.borrow();
        let mut state = ctx.state.lock().unwrap();
        compute_with_dispatch(py, &mut state, graph, &this, inputs)
    }

    /// Dispatch graph execution with MLTensor or MLDeviceTensor inputs/outputs
    ///
    /// Following the W3C WebNN MLTensor Explainer:
    /// https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md
    ///
    /// This method executes the graph with tensor inputs and writes results to output tensors.
    /// Supports both host tensors (MLTensor) and device tensors (MLDeviceTensor) for zero-copy execution.
    ///
    /// Args:
    ///     graph: The compiled MLGraph to execute
    ///     inputs: Dictionary mapping input names to MLTensor or MLDeviceTensor objects
    ///     outputs: Dictionary mapping output names to MLTensor or MLDeviceTensor objects
    ///
    /// Note:
    ///     When using MLDeviceTensor inputs/outputs, execution avoids host-device round-trips,
    ///     which is critical for iterative GenAI workloads like KV cache.
    #[pyo3(signature = (graph, inputs, outputs))]
    fn dispatch(
        this: Py<Self>,
        py: Python,
        graph: &mut PyMLGraph,
        inputs: &Bound<'_, PyDict>,
        outputs: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        for (_, value) in inputs.iter().chain(outputs.iter()) {
            if value.cast::<PyMLDeviceTensor>().is_ok() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "MLDeviceTensor is not supported on the rustnn MLContext execution path",
                ));
            }
        }

        let ctx = this.bind(py).borrow();
        let mut state = ctx.state.lock().unwrap();
        dispatch_with_ml_tensors(py, &mut state, graph, &this, inputs, outputs)
    }

    /// Resize a tensor's logical shape for dynamic-input graphs (KV cache, masks).
    ///
    /// Storage may be pre-allocated with `set_tensor_capacity`. See `rustnn/examples/smollm_mlcontext.rs`.
    fn resize_tensor(
        this: Py<Self>,
        _py: Python,
        tensor: &mut PyMLTensor,
        shape: Vec<u32>,
    ) -> PyResult<()> {
        tensor.check_destroyed()?;
        let ctx = this.bind(_py).borrow();
        let mut state = ctx.state.lock().unwrap();
        resize_rustnn_tensor(&mut state, &mut tensor.inner, &shape)
    }

    /// Pre-allocate tensor storage up to `max_shape` without changing the logical shape.
    fn set_tensor_capacity(
        this: Py<Self>,
        _py: Python,
        tensor: &mut PyMLTensor,
        max_shape: Vec<u32>,
    ) -> PyResult<()> {
        tensor.check_destroyed()?;
        let ctx = this.bind(_py).borrow();
        let mut state = ctx.state.lock().unwrap();
        set_rustnn_tensor_capacity(&mut state, &mut tensor.inner, &max_shape)
    }

    /// Convert graph to ONNX format
    ///
    /// Args:
    ///     graph: The MLGraph to convert
    ///     output_path: Path to save the ONNX model
    fn convert_to_onnx(&self, graph: &PyMLGraph, output_path: &str) -> PyResult<()> {
        let converter = rustnn::converters::OnnxConverter;
        let converted = converter.convert(&graph.graph_info).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("ONNX conversion failed: {}", e))
        })?;

        std::fs::write(output_path, &converted.data).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to write ONNX file: {}", e))
        })?;
        if let Some(weights) = &converted.weights_data {
            let sidecar = std::path::Path::new(output_path)
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join(rustnn::ONNX_EXTERNAL_WEIGHTS_FILENAME);
            std::fs::write(&sidecar, weights).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "Failed to write ONNX external weights file `{}`: {}",
                    sidecar.display(),
                    e
                ))
            })?;
        }

        Ok(())
    }

    /// Convert graph to CoreML format (macOS only)
    ///
    /// Args:
    ///     graph: The MLGraph to convert
    ///     output_path: Path to save the CoreML model
    #[cfg(target_os = "macos")]
    fn convert_to_coreml(&self, graph: &PyMLGraph, output_path: &str) -> PyResult<()> {
        let converter = rustnn::converters::CoremlMlProgramConverter;
        let converted = converter.convert(&graph.graph_info).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("CoreML conversion failed: {}", e))
        })?;

        std::fs::write(output_path, &converted.data).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to write CoreML file: {}", e))
        })?;

        Ok(())
    }

    /// Create a tensor (device-resident by default, per WebNN spec)
    ///
    /// Following the W3C WebNN specification:
    /// https://www.w3.org/TR/webnn/#dom-mlcontext-createtensor
    ///
    /// By default (readable=False, writable=False), creates a host-backed tensor.
    /// For true device-resident tensors with zero-copy execution, use create_device_tensor().
    ///
    /// Note: The spec intends device-resident tensors by default, but our implementation
    /// currently returns host-backed tensors for simplicity. We plan to add lazy device
    /// tensor materialization in a future version to fully match the spec.
    ///
    /// Args:
    ///     shape: Shape of the tensor
    ///     data_type: Data type string (e.g., "float32")
    ///     readable: If True, tensor data can be read back to CPU (default: False per spec)
    ///     writable: If True, tensor data can be written from CPU (default: False per spec)
    ///     exportable_to_gpu: If True, tensor can be used as GPU texture (default: False)
    ///
    /// Returns:
    ///     MLTensor: A new tensor with the specified properties
    ///
    /// Examples:
    ///     # Device-resident tensor (spec-compliant defaults)
    ///     tensor = context.create_tensor([2, 3], "float32")
    ///
    ///     # Host-accessible tensor (explicit flags)
    ///     host_tensor = context.create_tensor([2, 3], "float32", readable=True, writable=True)
    ///
    ///     # Convenience: use create_host_tensor() for host tensors
    ///     host_tensor = context.create_host_tensor([2, 3], "float32")
    #[pyo3(signature = (shape, data_type, readable=false, writable=false, exportable_to_gpu=false))]
    fn create_tensor(
        this: Py<Self>,
        py: Python,
        shape: Vec<u32>,
        data_type: &str,
        readable: bool,
        writable: bool,
        exportable_to_gpu: bool,
    ) -> PyResult<PyMLTensor> {
        let ctx = this.bind(py).borrow();
        let mut state = ctx.state.lock().unwrap();
        let inner = create_rustnn_tensor(&mut state, shape, data_type, readable, writable)?;
        Ok(PyMLTensor::from_rustnn(
            this.clone_ref(py),
            inner,
            exportable_to_gpu,
        ))
    }

    /// Convenience method for creating host-backed tensors (non-spec extension)
    ///
    /// This is equivalent to:
    ///   create_tensor(shape, data_type, readable=True, writable=True)
    ///
    /// Use this when you need to inspect or modify tensor contents from Python,
    /// such as for debugging, prototyping, or when your workflow requires host access.
    ///
    /// For production code with iterative workloads (like KV cache), prefer
    /// create_device_tensor() for optimal performance.
    ///
    /// Args:
    ///     shape: Shape of the tensor
    ///     data_type: Data type string (e.g., "float32")
    ///
    /// Returns:
    ///     MLTensor: A host-backed tensor (always readable and writable)
    ///
    /// Example:
    ///     # Quick and easy for prototyping
    ///     tensor = context.create_host_tensor([2, 3], "float32")
    ///     context.write_tensor(tensor, np.array([[1, 2, 3], [4, 5, 6]]))
    ///     data = context.read_tensor(tensor)
    #[pyo3(signature = (shape, data_type))]
    fn create_host_tensor(
        this: Py<Self>,
        py: Python,
        shape: Vec<u32>,
        data_type: &str,
    ) -> PyResult<PyMLTensor> {
        Self::create_tensor(this, py, shape, data_type, true, true, false)
    }

    #[pyo3(signature = (_graph, _shape, _data_type, _device=None))]
    fn create_device_tensor(
        &self,
        _graph: &PyMLGraph,
        _shape: Vec<usize>,
        _data_type: &str,
        _device: Option<&str>,
    ) -> PyResult<PyMLDeviceTensor> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "MLDeviceTensor is not supported on the rustnn MLContext execution path",
        ))
    }

    /// Read data from a tensor into a numpy array
    ///
    /// Follows the W3C WebNN MLTensor Explainer timeline model.
    ///
    /// Args:
    ///     tensor: The MLTensor to read from (must have readable=True)
    ///
    /// Returns:
    ///     numpy.ndarray: The tensor data as a numpy array
    ///
    /// Raises:
    ///     RuntimeError: If tensor is not readable or has been destroyed
    fn read_tensor<'py>(
        this: Py<Self>,
        py: Python<'py>,
        tensor: &PyMLTensor,
    ) -> PyResult<Bound<'py, PyAny>> {
        tensor.check_destroyed()?;
        if !tensor.readable() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Tensor is not readable (readable=false)",
            ));
        }
        let ctx = this.bind(py).borrow();
        let mut state = ctx.state.lock().unwrap();
        read_rustnn_tensor(py, &mut state, &tensor.inner)
    }

    /// Write data from a numpy array into a tensor
    ///
    /// Follows the W3C WebNN MLTensor Explainer timeline model.
    ///
    /// Args:
    ///     tensor: The MLTensor to write to (must have writable=True)
    ///     data: Numpy array or array-like data to write
    ///
    /// Raises:
    ///     RuntimeError: If tensor is not writable or has been destroyed
    ///     ValueError: If data shape doesn't match tensor shape
    fn write_tensor(
        this: Py<Self>,
        py: Python,
        tensor: &PyMLTensor,
        data: Bound<PyAny>,
    ) -> PyResult<()> {
        tensor.check_destroyed()?;
        if !tensor.writable() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Tensor is not writable (writable=false)",
            ));
        }
        let ctx = this.bind(py).borrow();
        let mut state = ctx.state.lock().unwrap();
        write_rustnn_tensor(py, &mut state, &tensor.inner, data)
    }

    /// Get power preference hint
    #[getter]
    fn power_preference(&self) -> String {
        self.power_preference.clone()
    }

    /// Check if GPU/NPU acceleration is available
    ///
    /// Returns:
    ///     bool: True if the platform can provide GPU or NPU resources
    ///
    /// Note:
    ///     This indicates platform capability, not a guarantee of device allocation.
    ///     The actual execution may still use CPU if needed.
    #[getter]
    fn accelerated(&self) -> bool {
        self.state.lock().unwrap().accelerated()
    }

    /// Get operation support limits for this context
    ///
    /// Returns a dictionary describing what operations and parameter types
    /// are supported by the backend implementation. This allows applications
    /// to query feature support and adapt their models accordingly.
    ///
    /// Returns:
    ///     dict: Dictionary with support limits for each operation
    ///
    /// Example:
    ///     >>> limits = context.op_support_limits()
    ///     >>> print(limits['preferredInputLayout'])
    ///     'nchw'
    ///     >>> print(limits['input']['dataTypes'])
    ///     ['float32', 'float16', 'int32', ...]
    fn op_support_limits(&self, py: Python) -> PyResult<Py<PyDict>> {
        let result = PyDict::new(py);

        // Helper function to create data type lists
        let create_float_types = || -> Vec<&str> { vec!["float32", "float16"] };

        let create_all_types = || -> Vec<&str> {
            vec![
                "float32", "float16", "int32", "uint32", "int8", "uint8", "int64", "uint64",
            ]
        };

        // Helper function to create rank range
        let create_rank_range = |py: Python| -> PyResult<Py<PyDict>> {
            let rank = PyDict::new(py);
            rank.set_item("min", 0)?;
            rank.set_item("max", 4)?; // Support up to 4D tensors
            Ok(rank.into())
        };

        // Helper function to create tensor limits
        let create_tensor_limits = |py: Python, float_only: bool| -> PyResult<Py<PyDict>> {
            let limits = PyDict::new(py);
            let types = if float_only {
                create_float_types()
            } else {
                create_all_types()
            };
            limits.set_item("dataTypes", types)?;
            limits.set_item("rankRange", create_rank_range(py)?)?;
            Ok(limits.into())
        };

        // Helper function to create single input limits
        let create_single_input_limits = |py: Python| -> PyResult<Py<PyDict>> {
            let limits = PyDict::new(py);
            limits.set_item("input", create_tensor_limits(py, true)?)?;
            limits.set_item("output", create_tensor_limits(py, true)?)?;
            Ok(limits.into())
        };

        // Helper function to create binary limits
        let create_binary_limits = |py: Python| -> PyResult<Py<PyDict>> {
            let limits = PyDict::new(py);
            limits.set_item("a", create_tensor_limits(py, true)?)?;
            limits.set_item("b", create_tensor_limits(py, true)?)?;
            limits.set_item("output", create_tensor_limits(py, true)?)?;
            Ok(limits.into())
        };

        // Top-level properties
        result.set_item("preferredInputLayout", "nchw")?;
        result.set_item("maxTensorByteLength", 4294967295u64)?; // 4GB max

        // Input, constant, output limits
        result.set_item("input", create_tensor_limits(py, false)?)?;
        result.set_item("constant", create_tensor_limits(py, false)?)?;
        result.set_item("output", create_tensor_limits(py, false)?)?;

        // Binary operations
        result.set_item("add", create_binary_limits(py)?)?;
        result.set_item("sub", create_binary_limits(py)?)?;
        result.set_item("mul", create_binary_limits(py)?)?;
        result.set_item("div", create_binary_limits(py)?)?;
        result.set_item("pow", create_binary_limits(py)?)?;
        result.set_item("matmul", create_binary_limits(py)?)?;

        // Comparison operations
        result.set_item("equal", create_binary_limits(py)?)?;
        result.set_item("greater", create_binary_limits(py)?)?;
        result.set_item("greaterOrEqual", create_binary_limits(py)?)?;
        result.set_item("lesser", create_binary_limits(py)?)?;
        result.set_item("lesserOrEqual", create_binary_limits(py)?)?;

        // Logical operations
        result.set_item("logicalAnd", create_binary_limits(py)?)?;
        result.set_item("logicalOr", create_binary_limits(py)?)?;
        result.set_item("logicalXor", create_binary_limits(py)?)?;
        result.set_item("logicalNot", create_single_input_limits(py)?)?;

        // Unary/activation operations
        result.set_item("relu", create_single_input_limits(py)?)?;
        result.set_item("sigmoid", create_single_input_limits(py)?)?;
        result.set_item("tanh", create_single_input_limits(py)?)?;
        result.set_item("softmax", create_single_input_limits(py)?)?;
        result.set_item("gelu", create_single_input_limits(py)?)?;
        result.set_item("elu", create_single_input_limits(py)?)?;
        result.set_item("leakyRelu", create_single_input_limits(py)?)?;
        result.set_item("hardSwish", create_single_input_limits(py)?)?;
        result.set_item("hardSigmoid", create_single_input_limits(py)?)?;
        result.set_item("clamp", create_single_input_limits(py)?)?;
        result.set_item("prelu", create_binary_limits(py)?)?;
        result.set_item("softplus", create_single_input_limits(py)?)?;
        result.set_item("softsign", create_single_input_limits(py)?)?;
        result.set_item("identity", create_single_input_limits(py)?)?;

        // Element-wise unary operations
        result.set_item("abs", create_single_input_limits(py)?)?;
        result.set_item("ceil", create_single_input_limits(py)?)?;
        result.set_item("floor", create_single_input_limits(py)?)?;
        result.set_item("neg", create_single_input_limits(py)?)?;
        result.set_item("sign", create_single_input_limits(py)?)?;
        result.set_item("reciprocal", create_single_input_limits(py)?)?;
        result.set_item("exp", create_single_input_limits(py)?)?;
        result.set_item("log", create_single_input_limits(py)?)?;
        result.set_item("sqrt", create_single_input_limits(py)?)?;
        result.set_item("erf", create_single_input_limits(py)?)?;

        // Trigonometric operations
        result.set_item("sin", create_single_input_limits(py)?)?;
        result.set_item("cos", create_single_input_limits(py)?)?;
        result.set_item("tan", create_single_input_limits(py)?)?;
        result.set_item("tanh", create_single_input_limits(py)?)?;

        // Type conversion
        let cast_limits = PyDict::new(py);
        cast_limits.set_item("input", create_tensor_limits(py, false)?)?;
        cast_limits.set_item("output", create_tensor_limits(py, false)?)?;
        result.set_item("cast", cast_limits)?;

        // Shape operations
        result.set_item("reshape", create_single_input_limits(py)?)?;
        result.set_item("transpose", create_single_input_limits(py)?)?;
        result.set_item("squeeze", create_single_input_limits(py)?)?;
        result.set_item("unsqueeze", create_single_input_limits(py)?)?;
        result.set_item("expand", create_single_input_limits(py)?)?;
        result.set_item("slice", create_single_input_limits(py)?)?;
        result.set_item("tile", create_single_input_limits(py)?)?;

        // Concat
        let concat_limits = PyDict::new(py);
        concat_limits.set_item("inputs", create_tensor_limits(py, true)?)?;
        concat_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("concat", concat_limits)?;

        // Split
        let split_limits = PyDict::new(py);
        split_limits.set_item("input", create_tensor_limits(py, true)?)?;
        split_limits.set_item("outputs", create_tensor_limits(py, true)?)?;
        result.set_item("split", split_limits)?;

        // Convolution
        let conv2d_limits = PyDict::new(py);
        conv2d_limits.set_item("input", create_tensor_limits(py, true)?)?;
        conv2d_limits.set_item("filter", create_tensor_limits(py, true)?)?;
        conv2d_limits.set_item("bias", create_tensor_limits(py, true)?)?;
        conv2d_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("conv2d", conv2d_limits)?;

        let conv_transpose_limits = PyDict::new(py);
        conv_transpose_limits.set_item("input", create_tensor_limits(py, true)?)?;
        conv_transpose_limits.set_item("filter", create_tensor_limits(py, true)?)?;
        conv_transpose_limits.set_item("bias", create_tensor_limits(py, true)?)?;
        conv_transpose_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("convTranspose2d", conv_transpose_limits)?;

        // Pooling
        let pool2d_limits = PyDict::new(py);
        pool2d_limits.set_item("input", create_tensor_limits(py, true)?)?;
        pool2d_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("averagePool2d", pool2d_limits.clone())?;
        result.set_item("maxPool2d", pool2d_limits)?;
        result.set_item("l2Pool2d", create_single_input_limits(py)?)?;

        // Normalization
        let batch_norm_limits = PyDict::new(py);
        batch_norm_limits.set_item("input", create_tensor_limits(py, true)?)?;
        batch_norm_limits.set_item("mean", create_tensor_limits(py, true)?)?;
        batch_norm_limits.set_item("variance", create_tensor_limits(py, true)?)?;
        batch_norm_limits.set_item("scale", create_tensor_limits(py, true)?)?;
        batch_norm_limits.set_item("bias", create_tensor_limits(py, true)?)?;
        batch_norm_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("batchNormalization", batch_norm_limits)?;

        let norm_limits = PyDict::new(py);
        norm_limits.set_item("input", create_tensor_limits(py, true)?)?;
        norm_limits.set_item("scale", create_tensor_limits(py, true)?)?;
        norm_limits.set_item("bias", create_tensor_limits(py, true)?)?;
        norm_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("instanceNormalization", norm_limits.clone())?;
        result.set_item("layerNormalization", norm_limits)?;

        // Reduction operations
        result.set_item("reduceSum", create_single_input_limits(py)?)?;
        result.set_item("reduceMean", create_single_input_limits(py)?)?;
        result.set_item("reduceMax", create_single_input_limits(py)?)?;
        result.set_item("reduceMin", create_single_input_limits(py)?)?;
        result.set_item("reduceProduct", create_single_input_limits(py)?)?;
        result.set_item("reduceL1", create_single_input_limits(py)?)?;
        result.set_item("reduceL2", create_single_input_limits(py)?)?;
        result.set_item("reduceLogSum", create_single_input_limits(py)?)?;
        result.set_item("reduceLogSumExp", create_single_input_limits(py)?)?;
        result.set_item("reduceSumSquare", create_single_input_limits(py)?)?;

        // GEMM
        let gemm_limits = PyDict::new(py);
        gemm_limits.set_item("a", create_tensor_limits(py, true)?)?;
        gemm_limits.set_item("b", create_tensor_limits(py, true)?)?;
        gemm_limits.set_item("c", create_tensor_limits(py, true)?)?;
        gemm_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("gemm", gemm_limits)?;

        // ArgMax/ArgMin
        let arg_limits = PyDict::new(py);
        arg_limits.set_item("input", create_tensor_limits(py, true)?)?;
        arg_limits.set_item("output", create_tensor_limits(py, false)?)?; // Outputs int64/uint64
        result.set_item("argMax", arg_limits.clone())?;
        result.set_item("argMin", arg_limits)?;

        // Gather operations
        let gather_limits = PyDict::new(py);
        gather_limits.set_item("input", create_tensor_limits(py, true)?)?;
        gather_limits.set_item("indices", create_tensor_limits(py, false)?)?;
        gather_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("gather", gather_limits)?;

        // Scatter operations
        let scatter_limits = PyDict::new(py);
        scatter_limits.set_item("input", create_tensor_limits(py, true)?)?;
        scatter_limits.set_item("indices", create_tensor_limits(py, false)?)?;
        scatter_limits.set_item("updates", create_tensor_limits(py, true)?)?;
        scatter_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("scatterElements", scatter_limits.clone())?;
        result.set_item("scatterND", scatter_limits)?;

        // Where
        let where_limits = PyDict::new(py);
        where_limits.set_item("condition", create_tensor_limits(py, false)?)?;
        where_limits.set_item("trueValue", create_tensor_limits(py, true)?)?;
        where_limits.set_item("falseValue", create_tensor_limits(py, true)?)?;
        where_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("where", where_limits)?;

        // Pad
        result.set_item("pad", create_single_input_limits(py)?)?;

        // Quantization
        let quant_limits = PyDict::new(py);
        quant_limits.set_item("input", create_tensor_limits(py, true)?)?;
        quant_limits.set_item("scale", create_tensor_limits(py, true)?)?;
        quant_limits.set_item("zeroPoint", create_tensor_limits(py, false)?)?;
        quant_limits.set_item("output", create_tensor_limits(py, false)?)?;
        result.set_item("quantizeLinear", quant_limits.clone())?;

        let dequant_limits = PyDict::new(py);
        dequant_limits.set_item("input", create_tensor_limits(py, false)?)?;
        dequant_limits.set_item("scale", create_tensor_limits(py, true)?)?;
        dequant_limits.set_item("zeroPoint", create_tensor_limits(py, false)?)?;
        dequant_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("dequantizeLinear", dequant_limits)?;

        // Triangular
        result.set_item("triangular", create_single_input_limits(py)?)?;

        Ok(result.into())
    }

    /// Return backend/feature diagnostics for this context
    ///
    /// Useful to understand which backend was selected, which runtime features
    /// were compiled into the wheel, and whether the selected backend is actually
    /// available (otherwise the fallback path returns zeros).
    fn backend_info(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let info = PyDict::new(py);
        let execution_compiled = cfg!(feature = "onnx-runtime");
        let coreml_compiled = cfg!(all(target_os = "macos", feature = "coreml-runtime"));
        let trtx_compiled = cfg!(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"));
        let litert_compiled = cfg!(feature = "litert-runtime");
        let cann_compiled = cfg!(any(feature = "cann-runtime", feature = "cann-runtime-mock"));

        info.set_item("accelerated_available", self.accelerated())?;
        info.set_item("device_type_requested", &self.device_type)?;
        info.set_item("backend_requested", &self.backend)?;
        info.set_item("compiled_features", {
            let compiled = PyDict::new(py);
            compiled.set_item("execution", execution_compiled)?;
            compiled.set_item("coreml", coreml_compiled)?;
            compiled.set_item("trtx", trtx_compiled)?;
            compiled.set_item("litert", litert_compiled)?;
            compiled.set_item("cann", cann_compiled)?;
            compiled
        })?;

        Ok(info.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "MLContext(accelerated={}, power='{}', backend='{}')",
            self.accelerated(),
            self.power_preference,
            self.backend
        )
    }
}

impl PyMLContext {
    pub(crate) fn compile_graph(&self, graph_info: GraphInfo) -> PyResult<usize> {
        self.state.lock().unwrap().compile_graph(graph_info)
    }

    fn new(
        power_preference: String,
        accelerated_requested: bool,
        device_type: String,
        backend: String,
    ) -> PyResult<Self> {
        let options = build_context_options(
            &power_preference,
            accelerated_requested,
            &device_type,
            &backend,
        )?;
        let state = ContextState::new(options)?;
        Ok(Self {
            power_preference,
            accelerated_requested,
            device_type,
            backend,
            state: Mutex::new(state),
        })
    }
}
