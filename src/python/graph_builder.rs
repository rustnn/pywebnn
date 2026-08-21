//! Graph builder for constructing WebNN computational graphs
//!
//! PyO3 macros generate unsafe code that triggers unsafe_op_in_unsafe_fn warnings.
//! This is expected behavior from the macro-generated code.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::too_many_arguments)]

use super::context_state::pack_numpy_to_4bit_bytes;
use super::graph::PyMLGraph;
use super::operand::{parse_data_type, PyMLOperand};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustnn::graph::{
    to_dimension_vector, ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::{
    MLArgMinMaxOptions, MLBatchNormalizationOptions, MLClampOptions, MLConv2dOptions,
    MLConvTranspose2dOptions, MLCumulativeSumOptions, MLDimension, MLEluOptions, MLGatherOptions,
    MLGemmOptions, MLGruCellOptions, MLGruOptions, MLHardSigmoidOptions,
    MLInstanceNormalizationOptions, MLLayerNormalizationOptions, MLLeakyReluOptions,
    MLLinearOptions, MLLstmCellOptions, MLLstmOptions, MLPadOptions, MLPool2dOptions,
    MLReduceOptions, MLResample2dOptions, MLReverseOptions, MLScatterOptions, MLSliceOptions,
    MLSplitOptions, MLSqueezeOptions, MLTransposeOptions, MLTriangularOptions, MLUnsqueezeOptions,
};
use rustnn::shape_inference::{
    broadcast_shapes, infer_equal_shape, infer_matmul_shape, infer_pool2d_shape,
    infer_resample2d_shape, infer_round_even_shape, validate_reshape,
};
use rustnn::validator::GraphValidator;
use rustnn::Operation;
use std::collections::HashMap;

fn infer_gather_nd_shape(input_shape: &[u32], indices_shape: &[u32]) -> Result<Vec<u32>, String> {
    if indices_shape.is_empty() {
        return Err("GatherND indices must have rank >= 1".to_string());
    }
    let k = indices_shape[indices_shape.len() - 1] as usize;
    if k > input_shape.len() {
        return Err(format!(
            "GatherND indices last dimension {} exceeds input rank {}",
            k,
            input_shape.len()
        ));
    }
    let mut shape = indices_shape[..indices_shape.len() - 1].to_vec();
    shape.extend_from_slice(&input_shape[k..]);
    Ok(shape)
}

fn recurrent_num_directions(direction: &str) -> u32 {
    if direction == "both" {
        2
    } else {
        1
    }
}

fn recurrent_batch_size(input_shape: &[u32]) -> u32 {
    match input_shape.len() {
        2 => input_shape[0],
        3 => input_shape[1],
        _ => 1,
    }
}

fn parse_pool_layout(layout: Option<&str>) -> PyResult<&'static str> {
    match layout.unwrap_or("nchw") {
        "nchw" => Ok("nchw"),
        "nhwc" => Ok("nhwc"),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid layout '{}', must be 'nchw' or 'nhwc'",
            other
        ))),
    }
}

fn make_pool2d_options(
    window_dimensions: Option<Vec<u32>>,
    strides: Vec<u32>,
    dilations: Vec<u32>,
    pads: Vec<u32>,
    layout: &str,
    output_shape_rounding: Option<&str>,
    output_sizes: Option<Vec<u32>>,
) -> MLPool2dOptions {
    MLPool2dOptions {
        label: String::new(),
        window_dimensions,
        padding: pads,
        strides,
        dilations,
        layout: layout.to_string(),
        output_shape_rounding: output_shape_rounding.unwrap_or("").to_string(),
        output_sizes,
    }
}

fn pad_options_value(
    py: Python<'_>,
    value: Option<Py<PyAny>>,
) -> PyResult<Option<serde_json::Value>> {
    match value {
        None => Ok(None),
        Some(obj) => {
            let v = obj.bind(py);
            if v.is_none() {
                return Ok(Some(serde_json::Value::Null));
            }
            if let Ok(f) = v.extract::<f64>() {
                return Ok(Some(serde_json::Value::from(f)));
            }
            if let Ok(i) = v.extract::<i64>() {
                return Ok(Some(serde_json::Value::from(i)));
            }
            if let Ok(u) = v.extract::<u64>() {
                return Ok(Some(serde_json::Value::from(u)));
            }
            if let Ok(s) = v.extract::<String>() {
                if let Ok(i) = s.parse::<i64>() {
                    return Ok(Some(serde_json::Value::from(i)));
                }
                if let Ok(f) = s.parse::<f64>() {
                    return Ok(Some(serde_json::Value::from(f)));
                }
                return Ok(Some(serde_json::Value::String(s)));
            }
            Err(pyo3::exceptions::PyTypeError::new_err(
                "pad value must be a number or string",
            ))
        }
    }
}

fn clamp_limit_to_json(
    py: Python<'_>,
    value: Option<Py<PyAny>>,
    default: f64,
) -> PyResult<serde_json::Value> {
    match value {
        None => Ok(serde_json::Value::from(default)),
        Some(obj) => {
            let v = obj.bind(py);
            if let Ok(i) = v.extract::<i64>() {
                return Ok(serde_json::Value::from(i));
            }
            if let Ok(u) = v.extract::<u64>() {
                return Ok(serde_json::Value::from(u));
            }
            if let Ok(f) = v.extract::<f64>() {
                return Ok(serde_json::Value::from(f));
            }
            Err(pyo3::exceptions::PyTypeError::new_err(
                "clamp limit must be a number",
            ))
        }
    }
}

fn clamp_limits_ordered(
    py: Python<'_>,
    min_value: Option<Py<PyAny>>,
    max_value: Option<Py<PyAny>>,
) -> PyResult<(serde_json::Value, serde_json::Value)> {
    let min_json = clamp_limit_to_json(py, min_value, f64::NEG_INFINITY)?;
    let max_json = clamp_limit_to_json(py, max_value, f64::INFINITY)?;
    let ordered = match (min_json.as_i64(), max_json.as_i64()) {
        (Some(min_i), Some(max_i)) => min_i <= max_i,
        _ => {
            let min_f = min_json.as_f64().unwrap_or(f64::NEG_INFINITY);
            let max_f = max_json.as_f64().unwrap_or(f64::INFINITY);
            min_f <= max_f
        }
    };
    if !ordered {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "clamp min_value ({}) must be <= max_value ({})",
            min_json, max_json
        )));
    }
    Ok((min_json, max_json))
}
#[pyclass(name = "MLGraphBuilder")]
pub struct PyMLGraphBuilder {
    context: Py<super::context::PyMLContext>,
    built: bool,
    operands: Vec<Operand>,
    operations: Vec<Operation>,
    input_operands: Vec<u32>,
    next_operand_id: u32,
    operand_map: HashMap<u32, PyMLOperand>,
    constant_data_map: HashMap<u32, ConstantData>,
}

#[pymethods]
impl PyMLGraphBuilder {
    #[new]
    fn new() -> PyResult<Self> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "MLGraphBuilder must be created via MLContext.create_graph_builder()",
        ))
    }

    /// Create an input operand
    ///
    /// Args:
    ///     name: Name of the input
    ///     shape: List of dimensions
    ///     data_type: Data type string (e.g., "float32")
    ///
    /// Returns:
    ///     MLOperand: The created input operand
    fn input(&mut self, name: String, shape: Vec<u32>, data_type: &str) -> PyResult<PyMLOperand> {
        let dtype = parse_data_type(data_type)?;
        let descriptor = OperandDescriptor {
            data_type: dtype,
            shape: to_dimension_vector(&shape),
            pending_permutation: Vec::new(),
        };

        let operand = Operand {
            descriptor: descriptor.clone(),
            kind: OperandKind::Input,
            name: Some(name.clone()),
        };

        let id = self.next_operand_id;
        self.operands.push(operand);
        self.input_operands.push(id);

        let py_operand = PyMLOperand::new(id, descriptor, OperandKind::Input, Some(name));
        self.operand_map.insert(id, py_operand.clone());
        self.next_operand_id += 1;

        Ok(py_operand)
    }

    /// Create a constant operand from numpy array
    ///
    /// Args:
    ///     value: NumPy array or Python list
    ///     shape: Optional shape override
    ///     data_type: Data type string (e.g., "float32")
    ///
    /// Returns:
    ///     MLOperand: The created constant operand
    #[pyo3(signature = (value, shape=None, data_type=None))]
    fn constant(
        &mut self,
        py: Python,
        value: &Bound<'_, PyAny>,
        shape: Option<Vec<u32>>,
        data_type: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        // Try to import numpy and convert to array
        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("asarray", (value,))?;

        // Get shape from array if not provided
        let actual_shape = if let Some(s) = shape {
            s
        } else {
            array.getattr("shape")?.extract::<Vec<u32>>()?
        };

        // Get dtype from array if not provided
        let actual_dtype = if let Some(dt) = data_type {
            parse_data_type(dt)?
        } else {
            let dtype_name: String = array.getattr("dtype")?.getattr("name")?.extract()?;
            parse_data_type(&dtype_name)?
        };

        let descriptor = OperandDescriptor {
            data_type: actual_dtype,
            shape: to_dimension_vector(&actual_shape),
            pending_permutation: Vec::new(),
        };

        // Convert array to bytes (nibble-packed for int4/uint4)
        let bytes = match actual_dtype {
            DataType::Int4 | DataType::Uint4 => pack_numpy_to_4bit_bytes(array, actual_dtype)?,
            _ => array.call_method0("tobytes")?.extract()?,
        };
        let constant_data = ConstantData {
            data: bytes,
            label: None,
        };

        let operand = Operand {
            descriptor: descriptor.clone(),
            kind: OperandKind::Constant,
            name: None,
        };

        let id = self.next_operand_id;
        self.operands.push(operand);
        self.constant_data_map.insert(id, constant_data);

        let py_operand = PyMLOperand::new(id, descriptor, OperandKind::Constant, None);
        self.operand_map.insert(id, py_operand.clone());
        self.next_operand_id += 1;

        Ok(py_operand)
    }

    // Binary operations

    /// Element-wise addition
    fn add(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.binary_op("add", a, b)
    }

    /// Element-wise subtraction
    fn sub(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.binary_op("sub", a, b)
    }

    /// Element-wise multiplication
    fn mul(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.binary_op("mul", a, b)
    }

    /// Element-wise division
    fn div(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.binary_op("div", a, b)
    }

    /// Element-wise power
    fn pow(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.binary_op("pow", a, b)
    }

    /// Element-wise maximum
    fn max(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.binary_op("max", a, b)
    }

    /// Element-wise minimum
    fn min(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.binary_op("min", a, b)
    }

    /// Matrix multiplication
    fn matmul(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        // Use proper matmul shape inference
        let output_shape = infer_matmul_shape(
            &a.descriptor.static_or_max_shape(),
            &b.descriptor.static_or_max_shape(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: a.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::Matmul {
            a: a.id,
            b: b.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// General Matrix Multiplication (GEMM)
    ///
    /// Computes: alpha * A' * B' + beta * C
    /// where A' and B' are optionally transposed versions of A and B.
    ///
    /// Args:
    ///     a: First input matrix (2D tensor)
    ///     b: Second input matrix (2D tensor)
    ///     c: Optional bias matrix (2D tensor, default: None)
    ///     alpha: Scalar multiplier for A*B (default: 1.0)
    ///     beta: Scalar multiplier for C (default: 1.0)
    ///     a_transpose: Whether to transpose A (default: False)
    ///     b_transpose: Whether to transpose B (default: False)
    ///
    ///Returns:
    ///     MLOperand: Output matrix [M, N]
    ///
    /// Example:
    ///     # Standard multiplication: Y = A * B
    ///     y = builder.gemm(a, b)
    ///
    ///     # With bias and transposed B: Y = A * B^T + C
    ///     y = builder.gemm(a, b, c=bias, b_transpose=True)
    #[pyo3(signature = (a, b, c=None, alpha=1.0, beta=1.0, a_transpose=false, b_transpose=false))]
    fn gemm(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        c: Option<&PyMLOperand>,
        alpha: f32,
        beta: f32,
        a_transpose: bool,
        b_transpose: bool,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_gemm_shape;

        let output_shape = infer_gemm_shape(
            &a.descriptor.static_or_max_shape(),
            &b.descriptor.static_or_max_shape(),
            a_transpose,
            b_transpose,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: a.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let c_operand_id = c.as_ref().map(|o| o.id);

        let gemm_options = MLGemmOptions {
            label: String::new(),
            c: c_operand_id,
            alpha: alpha as f64,
            beta: beta as f64,
            a_transpose,
            b_transpose,
        };

        self.push_op(Operation::Gemm {
            a: a.id,
            b: b.id,
            options: Some(gemm_options),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// 2D Convolution operation
    ///
    /// Args:
    ///     input: Input operand (4D tensor)
    ///     filter: Filter operand (4D tensor)
    ///     strides: Stride along each spatial axis (default: [1, 1])
    ///     dilations: Dilation along each spatial axis (default: [1, 1])
    ///     pads: Padding [begin_h, begin_w, end_h, end_w] (default: [0, 0, 0, 0])
    ///     groups: Number of groups (default: 1)
    ///     input_layout: Input layout "nchw" or "nhwc" (default: "nchw")
    ///     filter_layout: Filter layout "oihw", "hwio", "ohwi", or "ihwo" (default: "oihw")
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, filter, strides=None, dilations=None, pads=None, groups=None, input_layout=None, filter_layout=None, bias=None))]
    fn conv2d(
        &mut self,
        input: &PyMLOperand,
        filter: &PyMLOperand,
        strides: Option<Vec<u32>>,
        dilations: Option<Vec<u32>>,
        pads: Option<Vec<u32>>,
        groups: Option<u32>,
        input_layout: Option<&str>,
        filter_layout: Option<&str>,
        bias: Option<&PyMLOperand>,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_conv2d_shape;

        // Default values matching WebNN spec
        let strides = strides.unwrap_or_else(|| vec![1, 1]);
        let dilations = dilations.unwrap_or_else(|| vec![1, 1]);
        let pads = pads.unwrap_or_else(|| vec![0, 0, 0, 0]);
        let groups = groups.unwrap_or(1);

        let input_layout_s = match input_layout.unwrap_or("nchw") {
            "nchw" => "nchw",
            "nhwc" => "nhwc",
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid input_layout '{}', must be 'nchw' or 'nhwc'",
                    other
                )));
            }
        };

        let filter_layout_s = match filter_layout.unwrap_or("oihw") {
            "oihw" => "oihw",
            "hwio" => "hwio",
            "ohwi" => "ohwi",
            "ihwo" => "ihwo",
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid filter_layout '{}', must be 'oihw', 'hwio', 'ohwi', or 'ihwo'",
                    other
                )));
            }
        };

        let conv2d_options = MLConv2dOptions {
            label: String::new(),
            padding: pads,
            strides,
            dilations,
            groups,
            input_layout: input_layout_s.to_string(),
            filter_layout: filter_layout_s.to_string(),
            bias: bias.map(|b| b.id),
        };

        // Infer output shape
        let output_shape = infer_conv2d_shape(
            &input.descriptor.static_or_max_shape(),
            &filter.descriptor.static_or_max_shape(),
            &conv2d_options,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::Conv2d {
            input: input.id,
            filter: filter.id,
            options: Some(conv2d_options),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// 2D Transposed Convolution operation (deconvolution)
    #[pyo3(signature = (input, filter, strides=None, dilations=None, pads=None, output_padding=None, output_sizes=None, groups=None, input_layout=None, filter_layout=None, bias=None))]
    fn conv_transpose2d(
        &mut self,
        input: &PyMLOperand,
        filter: &PyMLOperand,
        strides: Option<Vec<u32>>,
        dilations: Option<Vec<u32>>,
        pads: Option<Vec<u32>>,
        output_padding: Option<Vec<u32>>,
        output_sizes: Option<Vec<u32>>,
        groups: Option<u32>,
        input_layout: Option<&str>,
        filter_layout: Option<&str>,
        bias: Option<&PyMLOperand>,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_conv_transpose2d_shape;

        // Default values matching WebNN spec
        let strides = strides.unwrap_or_else(|| vec![1, 1]);
        let dilations = dilations.unwrap_or_else(|| vec![1, 1]);
        let pads = pads.unwrap_or_else(|| vec![0, 0, 0, 0]);
        let output_padding = output_padding.unwrap_or_else(|| vec![0, 0]);
        let groups = groups.unwrap_or(1);

        let input_layout_s = match input_layout.unwrap_or("nchw") {
            "nchw" => "nchw",
            "nhwc" => "nhwc",
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid input_layout '{}', must be 'nchw' or 'nhwc'",
                    other
                )));
            }
        };

        let filter_layout_s = match filter_layout.unwrap_or("iohw") {
            "iohw" => "iohw",
            "hwoi" => "hwoi",
            "ohwi" => "ohwi",
            "oihw" => "oihw",
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid filter_layout '{}', must be 'iohw', 'hwoi', 'ohwi', or 'oihw'",
                    other
                )));
            }
        };

        let conv_t_options = MLConvTranspose2dOptions {
            label: String::new(),
            padding: pads,
            strides,
            dilations,
            output_padding,
            output_sizes: output_sizes.clone(),
            groups,
            input_layout: input_layout_s.to_string(),
            filter_layout: filter_layout_s.to_string(),
            bias: bias.map(|b| b.id),
        };

        // Infer output shape
        let output_shape = infer_conv_transpose2d_shape(
            &input.descriptor.static_or_max_shape(),
            &filter.descriptor.static_or_max_shape(),
            &conv_t_options,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::ConvTranspose2d {
            input: input.id,
            filter: filter.id,
            options: Some(conv_t_options),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// 2D Average Pooling operation
    ///
    /// Args:
    ///     input: Input operand (4D tensor)
    ///     window_dimensions: Size of the pooling window [height, width] (default: [1, 1])
    ///     strides: Stride along each spatial axis (default: [1, 1])
    ///     dilations: Dilation along each spatial axis (default: [1, 1])
    ///     pads: Padding [begin_h, begin_w, end_h, end_w] (default: [0, 0, 0, 0])
    ///     layout: Input layout "nchw" or "nhwc" (default: "nchw")
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, window_dimensions=None, strides=None, dilations=None, pads=None, layout=None, output_shape_rounding=None, output_sizes=None))]
    fn average_pool2d(
        &mut self,
        input: &PyMLOperand,
        window_dimensions: Option<Vec<u32>>,
        strides: Option<Vec<u32>>,
        dilations: Option<Vec<u32>>,
        pads: Option<Vec<u32>>,
        layout: Option<&str>,
        output_shape_rounding: Option<&str>,
        output_sizes: Option<Vec<u32>>,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_pool2d_shape;

        // Default values matching WebNN spec (window_dimensions omitted => full spatial size)
        let strides = strides.unwrap_or_else(|| vec![1, 1]);
        let dilations = dilations.unwrap_or_else(|| vec![1, 1]);
        let pads = pads.unwrap_or_else(|| vec![0, 0, 0, 0]);
        let layout_s = parse_pool_layout(layout)?;

        let pool_opts = make_pool2d_options(
            window_dimensions,
            strides,
            dilations,
            pads,
            layout_s,
            output_shape_rounding,
            output_sizes,
        );

        // Infer output shape
        let output_shape = infer_pool2d_shape(&input.descriptor.static_or_max_shape(), &pool_opts)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::AveragePool2d {
            input: input.id,
            options: Some(pool_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// 2D Max Pooling operation
    ///
    /// Args:
    ///     input: Input operand (4D tensor)
    ///     window_dimensions: Size of the pooling window [height, width] (default: [1, 1])
    ///     strides: Stride along each spatial axis (default: [1, 1])
    ///     dilations: Dilation along each spatial axis (default: [1, 1])
    ///     pads: Padding [begin_h, begin_w, end_h, end_w] (default: [0, 0, 0, 0])
    ///     layout: Input layout "nchw" or "nhwc" (default: "nchw")
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, window_dimensions=None, strides=None, dilations=None, pads=None, layout=None, output_shape_rounding=None, output_sizes=None))]
    fn max_pool2d(
        &mut self,
        input: &PyMLOperand,
        window_dimensions: Option<Vec<u32>>,
        strides: Option<Vec<u32>>,
        dilations: Option<Vec<u32>>,
        pads: Option<Vec<u32>>,
        layout: Option<&str>,
        output_shape_rounding: Option<&str>,
        output_sizes: Option<Vec<u32>>,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_pool2d_shape;

        // Default values matching WebNN spec (window_dimensions omitted => full spatial size)
        let strides = strides.unwrap_or_else(|| vec![1, 1]);
        let dilations = dilations.unwrap_or_else(|| vec![1, 1]);
        let pads = pads.unwrap_or_else(|| vec![0, 0, 0, 0]);
        let layout_s = parse_pool_layout(layout)?;

        let pool_opts = make_pool2d_options(
            window_dimensions,
            strides,
            dilations,
            pads,
            layout_s,
            output_shape_rounding,
            output_sizes,
        );

        // Infer output shape
        let output_shape = infer_pool2d_shape(&input.descriptor.static_or_max_shape(), &pool_opts)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::MaxPool2d {
            input: input.id,
            options: Some(pool_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// 2D L2 Pooling operation
    #[pyo3(signature = (input, window_dimensions=None, strides=None, dilations=None, pads=None, layout=None, output_shape_rounding=None, output_sizes=None))]
    fn l2_pool2d(
        &mut self,
        input: &PyMLOperand,
        window_dimensions: Option<Vec<u32>>,
        strides: Option<Vec<u32>>,
        dilations: Option<Vec<u32>>,
        pads: Option<Vec<u32>>,
        layout: Option<&str>,
        output_shape_rounding: Option<&str>,
        output_sizes: Option<Vec<u32>>,
    ) -> PyResult<PyMLOperand> {
        let strides = strides.unwrap_or_else(|| vec![1, 1]);
        let dilations = dilations.unwrap_or_else(|| vec![1, 1]);
        let pads = pads.unwrap_or_else(|| vec![0, 0, 0, 0]);
        let layout_s = parse_pool_layout(layout)?;

        let pool_opts = make_pool2d_options(
            window_dimensions,
            strides,
            dilations,
            pads,
            layout_s,
            output_shape_rounding,
            output_sizes,
        );

        let output_shape = infer_pool2d_shape(&input.descriptor.static_or_max_shape(), &pool_opts)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::L2Pool2d {
            input: input.id,
            options: Some(pool_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Global Average Pooling operation
    ///
    /// Args:
    ///     input: Input operand (4D tensor)
    ///     layout: Input layout "nchw" or "nhwc" (default: "nchw")
    ///
    /// Returns:
    ///     MLOperand: The output operand with spatial dimensions reduced to 1x1
    #[pyo3(signature = (input, layout=None))]
    fn global_average_pool(
        &mut self,
        input: &PyMLOperand,
        layout: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::{infer_global_pool_shape, InputLayout};

        let layout_enum = match layout.unwrap_or("nchw") {
            "nchw" => InputLayout::Nchw,
            "nhwc" => InputLayout::Nhwc,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid layout '{}', must be 'nchw' or 'nhwc'",
                    other
                )));
            }
        };

        let output_shape =
            infer_global_pool_shape(&input.descriptor.static_or_max_shape(), layout_enum)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let pool_opts = MLPool2dOptions {
            label: String::new(),
            window_dimensions: None,
            padding: vec![],
            strides: vec![],
            dilations: vec![],
            layout: layout.unwrap_or("nchw").to_string(),
            output_shape_rounding: String::new(),
            output_sizes: None,
        };

        self.push_op(Operation::GlobalAveragePool {
            input: input.id,
            options: Some(pool_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Global Max Pooling operation
    ///
    /// Args:
    ///     input: Input operand (4D tensor)
    ///     layout: Input layout "nchw" or "nhwc" (default: "nchw")
    ///
    /// Returns:
    ///     MLOperand: The output operand with spatial dimensions reduced to 1x1
    #[pyo3(signature = (input, layout=None))]
    fn global_max_pool(
        &mut self,
        input: &PyMLOperand,
        layout: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::{infer_global_pool_shape, InputLayout};

        let layout_enum = match layout.unwrap_or("nchw") {
            "nchw" => InputLayout::Nchw,
            "nhwc" => InputLayout::Nhwc,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid layout '{}', must be 'nchw' or 'nhwc'",
                    other
                )));
            }
        };

        let output_shape =
            infer_global_pool_shape(&input.descriptor.static_or_max_shape(), layout_enum)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let pool_opts = MLPool2dOptions {
            label: String::new(),
            window_dimensions: None,
            padding: vec![],
            strides: vec![],
            dilations: vec![],
            layout: layout.unwrap_or("nchw").to_string(),
            output_shape_rounding: String::new(),
            output_sizes: None,
        };

        self.push_op(Operation::GlobalMaxPool {
            input: input.id,
            options: Some(pool_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    // Normalization operations

    /// Batch Normalization operation
    #[pyo3(signature = (input, mean, variance, scale=None, bias=None, epsilon=1e-5, axis=1))]
    fn batch_normalization(
        &mut self,
        input: &PyMLOperand,
        mean: &PyMLOperand,
        variance: &PyMLOperand,
        scale: Option<&PyMLOperand>,
        bias: Option<&PyMLOperand>,
        epsilon: f32,
        axis: u32,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_batch_normalization_shape;

        // Infer output shape (same as input for batch normalization)
        let output_shape = infer_batch_normalization_shape(&input.descriptor.static_or_max_shape())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Create output descriptor
        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let bn_options = MLBatchNormalizationOptions {
            label: String::new(),
            scale: scale.map(|s| s.id),
            bias: bias.map(|b| b.id),
            axis,
            epsilon: epsilon as f64,
        };

        self.push_op(Operation::BatchNormalization {
            input: input.id,
            mean: mean.id,
            variance: variance.id,
            options: Some(bn_options),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Instance Normalization operation
    #[pyo3(signature = (input, scale=None, bias=None, epsilon=1e-5, layout=None))]
    fn instance_normalization(
        &mut self,
        input: &PyMLOperand,
        scale: Option<&PyMLOperand>,
        bias: Option<&PyMLOperand>,
        epsilon: f32,
        layout: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_instance_normalization_shape;

        // Infer output shape (same as input for instance normalization)
        let output_shape =
            infer_instance_normalization_shape(&input.descriptor.static_or_max_shape())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Create output descriptor
        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let inst_options = MLInstanceNormalizationOptions {
            label: String::new(),
            scale: scale.map(|s| s.id),
            bias: bias.map(|b| b.id),
            epsilon: epsilon as f64,
            layout: layout.unwrap_or("nchw").to_string(),
        };

        self.push_op(Operation::InstanceNormalization {
            input: input.id,
            options: Some(inst_options),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Layer Normalization operation
    #[pyo3(signature = (input, scale=None, bias=None, epsilon=1e-5, axes=None))]
    fn layer_normalization(
        &mut self,
        input: &PyMLOperand,
        scale: Option<&PyMLOperand>,
        bias: Option<&PyMLOperand>,
        epsilon: f32,
        axes: Option<Vec<u32>>,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_layer_normalization_shape;

        // Infer output shape (same as input for layer normalization)
        let output_shape = infer_layer_normalization_shape(&input.descriptor.static_or_max_shape())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Create output descriptor
        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        // When axes are omitted, infer them from scale/bias rank so ONNX axis aligns
        // with X.shape[axis:] for broadcast-compatible operands.
        let norm_axes: Vec<u32> = if let Some(axes) = axes {
            axes
        } else {
            let input_rank = input.descriptor.static_or_max_shape().len();
            let param_rank = scale
                .map(|s| s.descriptor.static_or_max_shape().len())
                .or_else(|| bias.map(|b| b.descriptor.static_or_max_shape().len()));

            match param_rank {
                Some(rank) if rank > 0 && rank <= input_rank => ((input_rank - rank)..input_rank)
                    .map(|i| i as u32)
                    .collect(),
                // WebNN default: normalize over axes 1..rank-1 (all non-batch dimensions).
                _ => (1..input_rank as u32).collect(),
            }
        };

        let layer_options = MLLayerNormalizationOptions {
            label: String::new(),
            scale: scale.map(|s| s.id),
            bias: bias.map(|b| b.id),
            axes: Some(norm_axes),
            epsilon: epsilon as f64,
        };

        self.push_op(Operation::LayerNormalization {
            input: input.id,
            options: Some(layer_options),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    // Unary operations

    /// ReLU activation
    fn relu(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("relu", x)
    }

    /// Sigmoid activation
    fn sigmoid(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("sigmoid", x)
    }

    /// Tanh activation
    fn tanh(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("tanh", x)
    }

    /// Softmax activation
    ///
    /// Args:
    ///     x: Input operand
    ///     axis: Axis along which to normalize
    ///
    /// Returns:
    ///     MLOperand: Output operand with the same shape as input
    fn softmax(&mut self, x: &PyMLOperand, axis: u32) -> PyResult<PyMLOperand> {
        let rank = x.descriptor.static_or_max_shape().len() as u32;
        if rank == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "softmax input must have rank at least 1",
            ));
        }
        if axis >= rank {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Axis {} out of bounds for rank {}",
                axis, rank
            )));
        }

        let output_descriptor = x.descriptor.clone();

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::Softmax {
            input: x.id,
            axis,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    // Element-wise operations - Basic math

    /// Element-wise absolute value
    fn abs(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("abs", x)
    }

    /// Element-wise ceiling (round up)
    fn ceil(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("ceil", x)
    }

    /// Element-wise floor (round down)
    fn floor(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("floor", x)
    }

    /// Element-wise round-to-nearest-even (banker's rounding)
    fn round_even(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        let output_shape = infer_round_even_shape(&x.descriptor.static_or_max_shape())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: x.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::RoundEven {
            input: x.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Element-wise negation
    fn neg(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("neg", x)
    }

    /// Element-wise sign (-1, 0, 1)
    fn sign(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("sign", x)
    }

    // Element-wise operations - Exponential and logarithmic

    /// Element-wise natural exponential (e^x)
    fn exp(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("exp", x)
    }

    /// Element-wise natural logarithm
    fn log(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("log", x)
    }

    /// Element-wise square root
    fn sqrt(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("sqrt", x)
    }

    /// Element-wise reciprocal (1/x)
    fn reciprocal(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("reciprocal", x)
    }

    // Element-wise operations - Trigonometric

    /// Element-wise sine
    fn sin(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("sin", x)
    }

    /// Element-wise cosine
    fn cos(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("cos", x)
    }

    /// Element-wise tangent
    fn tan(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("tan", x)
    }

    // Element-wise operations - Special functions

    /// Element-wise error function
    fn erf(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("erf", x)
    }

    /// Identity operation (returns input unchanged)
    fn identity(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("identity", x)
    }

    // Logic operations

    /// Element-wise equality comparison
    fn equal(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_equal_shape;

        let output_shape = infer_equal_shape(
            &a.descriptor.static_or_max_shape(),
            &b.descriptor.static_or_max_shape(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8, // Boolean output as uint8
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::Equal {
            a: a.id,
            b: b.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Element-wise greater than comparison
    fn greater(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_greater_shape;

        let output_shape = infer_greater_shape(
            &a.descriptor.static_or_max_shape(),
            &b.descriptor.static_or_max_shape(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::Greater {
            a: a.id,
            b: b.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Element-wise greater than or equal comparison
    fn greater_or_equal(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_greater_or_equal_shape;

        let output_shape = infer_greater_or_equal_shape(
            &a.descriptor.static_or_max_shape(),
            &b.descriptor.static_or_max_shape(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::GreaterOrEqual {
            a: a.id,
            b: b.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Element-wise less than comparison
    fn lesser(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_lesser_shape;

        let output_shape = infer_lesser_shape(
            &a.descriptor.static_or_max_shape(),
            &b.descriptor.static_or_max_shape(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::Lesser {
            a: a.id,
            b: b.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Element-wise less than or equal comparison
    fn lesser_or_equal(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_lesser_or_equal_shape;

        let output_shape = infer_lesser_or_equal_shape(
            &a.descriptor.static_or_max_shape(),
            &b.descriptor.static_or_max_shape(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::LesserOrEqual {
            a: a.id,
            b: b.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Element-wise not-equal comparison
    fn not_equal(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        let output_shape = infer_equal_shape(
            &a.descriptor.static_or_max_shape(),
            &b.descriptor.static_or_max_shape(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::NotEqual {
            a: a.id,
            b: b.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Element-wise logical NOT
    fn logical_not(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_logical_not_shape;

        let output_shape = infer_logical_not_shape(&x.descriptor.static_or_max_shape())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::LogicalNot {
            input: x.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Element-wise logical AND
    fn logical_and(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_logical_and_shape;

        let output_shape = infer_logical_and_shape(
            &a.descriptor.static_or_max_shape(),
            &b.descriptor.static_or_max_shape(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::LogicalAnd {
            a: a.id,
            b: b.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Element-wise logical OR
    fn logical_or(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_logical_or_shape;

        let output_shape = infer_logical_or_shape(
            &a.descriptor.static_or_max_shape(),
            &b.descriptor.static_or_max_shape(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::LogicalOr {
            a: a.id,
            b: b.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Element-wise logical XOR
    fn logical_xor(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_logical_xor_shape;

        let output_shape = infer_logical_xor_shape(
            &a.descriptor.static_or_max_shape(),
            &b.descriptor.static_or_max_shape(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::LogicalXor {
            a: a.id,
            b: b.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// DequantizeLinear operation
    /// Converts quantized integer values to floating-point representation
    /// Formula: output = (input - zeroPoint) * scale
    fn dequantize_linear(
        &mut self,
        input: &PyMLOperand,
        scale: &PyMLOperand,
        zero_point: &PyMLOperand,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_dequantize_linear_shape;

        let output_shape = infer_dequantize_linear_shape(&input.descriptor.static_or_max_shape())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // WebNN: output dtype matches scale (float16 or float32).
        let output_descriptor = OperandDescriptor {
            data_type: scale.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::DequantizeLinear {
            input: input.id,
            scale: scale.id,
            zero_point: Some(zero_point.id),
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// QuantizeLinear operation
    /// Converts floating-point values to quantized integer representation
    /// Formula: output = input / scale + zeroPoint
    fn quantize_linear(
        &mut self,
        input: &PyMLOperand,
        scale: &PyMLOperand,
        zero_point: &PyMLOperand,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_quantize_linear_shape;

        let output_shape = infer_quantize_linear_shape(&input.descriptor.static_or_max_shape())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Output data type matches zero_point's data type (typically int8 or uint8)
        let output_descriptor = OperandDescriptor {
            data_type: zero_point.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::QuantizeLinear {
            input: input.id,
            scale: scale.id,
            zero_point: Some(zero_point.id),
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Reshape operation
    fn reshape(&mut self, x: &PyMLOperand, new_shape: Vec<u32>) -> PyResult<PyMLOperand> {
        // Validate that reshape is possible
        validate_reshape(&x.descriptor.static_or_max_shape(), &new_shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: x.descriptor.data_type,
            shape: to_dimension_vector(&new_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let reshape_new_shape: Vec<MLDimension> =
            new_shape.iter().map(|&d| MLDimension::Static(d)).collect();

        self.push_op(Operation::Reshape {
            input: x.id,
            new_shape: reshape_new_shape,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    // Reduction operations

    /// Reduce Sum operation
    ///
    /// Reduces the input tensor by summing elements along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_sum(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceSum", input, axes, keep_dimensions)
    }

    /// Reduce Mean operation
    ///
    /// Reduces the input tensor by computing the arithmetic mean along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_mean(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceMean", input, axes, keep_dimensions)
    }

    /// Reduce Max operation
    ///
    /// Reduces the input tensor by computing the maximum value along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_max(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceMax", input, axes, keep_dimensions)
    }

    /// Reduce Min operation
    ///
    /// Reduces the input tensor by computing the minimum value along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_min(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceMin", input, axes, keep_dimensions)
    }

    /// Reduce Product operation
    ///
    /// Reduces the input tensor by computing the product of elements along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_product(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceProduct", input, axes, keep_dimensions)
    }

    /// Reduce L1 operation
    ///
    /// Reduces the input tensor by computing the L1 norm (sum of absolute values) along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_l1(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceL1", input, axes, keep_dimensions)
    }

    /// Reduce L2 operation
    ///
    /// Reduces the input tensor by computing the L2 norm (Euclidean norm) along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_l2(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceL2", input, axes, keep_dimensions)
    }

    /// Reduce Log Sum operation
    ///
    /// Reduces the input tensor by computing the natural logarithm of the sum along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_log_sum(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceLogSum", input, axes, keep_dimensions)
    }

    /// Reduce Log Sum Exp operation
    ///
    /// Reduces the input tensor by computing the log of the sum of exponentials along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_log_sum_exp(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceLogSumExp", input, axes, keep_dimensions)
    }

    /// Reduce Sum Square operation
    ///
    /// Reduces the input tensor by computing the sum of squares along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_sum_square(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceSumSquare", input, axes, keep_dimensions)
    }

    // Tensor manipulation operations

    /// Transpose operation
    ///
    /// Reorders the dimensions of a tensor according to a permutation.
    /// If no permutation is provided, reverses the dimensions.
    ///
    /// Args:
    ///     input: Input operand
    ///     permutation: Optional permutation of dimensions (default: reverse dimensions)
    ///
    /// Returns:
    ///     MLOperand: The transposed output operand
    #[pyo3(signature = (input, permutation=None))]
    fn transpose(
        &mut self,
        input: &PyMLOperand,
        permutation: Option<Vec<u32>>,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_transpose_shape;

        // Infer output shape
        let output_shape = infer_transpose_shape(
            &input.descriptor.static_or_max_shape(),
            permutation.as_deref(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let transpose_opts = MLTransposeOptions {
            label: String::new(),
            permutation: permutation.unwrap_or_default(),
        };

        self.push_op(Operation::Transpose {
            input: input.id,
            options: Some(transpose_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Concat operation
    ///
    /// Concatenates multiple tensors along a specified axis.
    ///
    /// Args:
    ///     inputs: List of input operands to concatenate
    ///     axis: Axis along which to concatenate
    ///
    /// Returns:
    ///     MLOperand: The concatenated output operand
    fn concat(&mut self, inputs: Vec<PyMLOperand>, axis: u32) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_concat_shape;

        if inputs.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Concat requires at least one input",
            ));
        }

        // Collect input shapes
        let input_shapes: Vec<Vec<u32>> = inputs
            .iter()
            .map(|op| op.descriptor.static_or_max_shape())
            .collect();

        // Infer output shape
        let output_shape = infer_concat_shape(&input_shapes, axis)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: inputs[0].descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        // Collect input IDs
        let input_ids: Vec<u32> = inputs.iter().map(|op| op.id).collect();

        self.push_op(Operation::Concat {
            inputs: input_ids,
            axis,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Slice operation
    ///
    /// Extracts a contiguous sub-tensor from the input.
    ///
    /// Args:
    ///     input: Input operand
    ///     starts: Starting indices for each dimension
    ///     sizes: Size of the slice for each dimension
    ///     strides: Optional stride values for each dimension (defaults to 1)
    ///
    /// Returns:
    ///     MLOperand: The sliced output operand
    #[pyo3(signature = (input, starts, sizes, strides=None))]
    fn slice(
        &mut self,
        input: &PyMLOperand,
        starts: Vec<u32>,
        sizes: Vec<u32>,
        strides: Option<Vec<i32>>,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_slice_shape;

        let input_rank = input.descriptor.static_or_max_shape().len();
        if starts.len() != sizes.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "slice() requires starts.len() == sizes.len() (got {} and {})",
                starts.len(),
                sizes.len()
            )));
        }
        // WebNN: for 0D input, starts and sizes must be length 0 (empty); that is a valid no-op.
        if (starts.is_empty() || sizes.is_empty()) && input_rank != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "slice() requires non-empty starts and sizes when input rank is not 0",
            ));
        }
        // 0D no-op: output shape is same as input

        let strides_vec: Vec<u32> = strides
            .as_ref()
            .map(|s| s.iter().map(|&x| x as u32).collect())
            .unwrap_or_default();
        let strides_arg = if strides_vec.is_empty() {
            None
        } else {
            Some(strides_vec.as_slice())
        };

        // Infer output shape (for 0D with empty starts/sizes, use input shape; else infer)
        let output_shape = if input_rank == 0 && starts.is_empty() {
            input.descriptor.static_or_max_shape()
        } else {
            infer_slice_shape(
                &input.descriptor.static_or_max_shape(),
                &starts,
                &sizes,
                strides_arg,
            )
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        };

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let sizes_dims: Vec<MLDimension> = sizes.iter().copied().map(MLDimension::Static).collect();

        let slice_opts = MLSliceOptions {
            label: String::new(),
            strides: strides_vec,
        };

        self.push_op(Operation::Slice {
            input: input.id,
            starts,
            sizes: sizes_dims,
            options: Some(slice_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Expand operation
    ///
    /// Broadcasts a tensor to a larger shape. Dimensions of size 1 can be expanded to larger sizes.
    ///
    /// Args:
    ///     input: Input operand
    ///     new_shape: Target shape for expansion
    ///
    /// Returns:
    ///     MLOperand: The expanded output operand
    fn expand(&mut self, input: &PyMLOperand, new_shape: Vec<u32>) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_expand_shape;

        // Infer output shape
        let output_shape = infer_expand_shape(&input.descriptor.static_or_max_shape(), &new_shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let expand_new_shape: Vec<MLDimension> =
            new_shape.iter().map(|&d| MLDimension::Static(d)).collect();

        self.push_op(Operation::Expand {
            input: input.id,
            new_shape: expand_new_shape,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Gather operation
    ///
    /// Gathers values from input tensor along an axis according to indices.
    ///
    /// Args:
    ///     input: Input operand
    ///     indices: Indices tensor
    ///     axis: Axis along which to gather (default: 0)
    ///
    /// Returns:
    ///     MLOperand: The gathered output operand
    #[pyo3(signature = (input, indices, axis=0))]
    fn gather(
        &mut self,
        input: &PyMLOperand,
        indices: &PyMLOperand,
        axis: u32,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_gather_shape;

        // Infer output shape
        let output_shape = infer_gather_shape(
            &input.descriptor.static_or_max_shape(),
            &indices.descriptor.static_or_max_shape(),
            axis,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let gather_opts = MLGatherOptions {
            label: String::new(),
            axis,
        };

        self.push_op(Operation::Gather {
            input: input.id,
            indices: indices.id,
            batch_dimensions: None,
            options: Some(gather_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Gather elements operation (output shape equals indices shape)
    #[pyo3(signature = (input, indices, axis=0))]
    fn gather_elements(
        &mut self,
        input: &PyMLOperand,
        indices: &PyMLOperand,
        axis: u32,
    ) -> PyResult<PyMLOperand> {
        let output_shape = indices.descriptor.static_or_max_shape();

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let gather_opts = MLGatherOptions {
            label: String::new(),
            axis,
        };

        self.push_op(Operation::GatherElements {
            input: input.id,
            indices: indices.id,
            batch_dimensions: None,
            options: Some(gather_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Gather ND operation
    fn gather_nd(&mut self, input: &PyMLOperand, indices: &PyMLOperand) -> PyResult<PyMLOperand> {
        let output_shape = infer_gather_nd_shape(
            &input.descriptor.static_or_max_shape(),
            &indices.descriptor.static_or_max_shape(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::GatherND {
            input: input.id,
            indices: indices.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Split operation
    ///
    /// Splits a tensor into multiple sub-tensors along an axis.
    ///
    /// Args:
    ///     input: Input operand
    ///     splits: Either number of equal splits (int) or list of split sizes
    ///     axis: Axis along which to split (default: 0)
    ///
    /// Returns:
    ///     List[MLOperand]: List of output operands
    #[pyo3(signature = (input, splits, axis=0))]
    fn split(
        &mut self,
        _py: Python,
        input: &PyMLOperand,
        splits: &Bound<'_, PyAny>,
        axis: u32,
    ) -> PyResult<Vec<PyMLOperand>> {
        use rustnn::shape_inference::{infer_split_shapes, SplitSpec};

        // Determine split specification
        let split_spec = if let Ok(count) = splits.extract::<u32>() {
            SplitSpec::Count(count)
        } else if let Ok(sizes) = splits.extract::<Vec<u32>>() {
            SplitSpec::Sizes(sizes)
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "splits must be either an integer or a list of integers",
            ));
        };

        // Infer output shapes
        let output_shapes =
            infer_split_shapes(&input.descriptor.static_or_max_shape(), &split_spec, axis)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Create output operands
        let mut py_operands = Vec::new();
        let mut output_ids = Vec::new();

        for output_shape in &output_shapes {
            let output_descriptor = OperandDescriptor {
                data_type: input.descriptor.data_type,
                shape: to_dimension_vector(output_shape),
                pending_permutation: Vec::new(),
            };

            let output_id = self.next_operand_id;
            self.next_operand_id += 1;

            let output_operand = Operand {
                descriptor: output_descriptor.clone(),
                kind: OperandKind::Output,
                name: None,
            };
            self.operands.push(output_operand);

            let py_operand =
                PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
            self.operand_map.insert(output_id, py_operand.clone());
            py_operands.push(py_operand);
            output_ids.push(output_id);
        }

        let split_opts = MLSplitOptions {
            label: String::new(),
            axis,
        };

        let (splits, split_equal_parts) = match &split_spec {
            SplitSpec::Count(n) => (Vec::new(), Some(*n)),
            SplitSpec::Sizes(sizes) => (sizes.clone(), None),
        };

        self.push_op(Operation::Split {
            input: input.id,
            splits,
            split_equal_parts,
            options: Some(split_opts),
            outputs: output_ids,
        });

        Ok(py_operands)
    }

    /// Where operation
    ///
    /// Selects elements from trueValue or falseValue based on condition.
    /// All inputs are broadcast to a common shape.
    ///
    /// Args:
    ///     condition: Boolean condition tensor
    ///     true_value: Values to select when condition is true
    ///     false_value: Values to select when condition is false
    ///
    /// Returns:
    ///     MLOperand: The output operand
    fn where_(
        &mut self,
        condition: &PyMLOperand,
        true_value: &PyMLOperand,
        false_value: &PyMLOperand,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_where_shape;

        // Infer output shape
        let output_shape = infer_where_shape(
            &condition.descriptor.static_or_max_shape(),
            &true_value.descriptor.static_or_max_shape(),
            &false_value.descriptor.static_or_max_shape(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: true_value.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::Where {
            condition: condition.id,
            true_value: true_value.id,
            false_value: false_value.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Pad operation
    ///
    /// Adds padding around the input tensor.
    ///
    /// Args:
    ///     input: Input operand
    ///     padding: Padding values [begin_0, begin_1, ..., end_0, end_1, ...]
    ///     mode: Padding mode ("constant", "edge", "reflection", "symmetric") (default: "constant")
    ///     value: Padding value for constant mode (default: 0.0)
    ///
    /// Returns:
    ///     MLOperand: The padded output operand
    #[pyo3(signature = (input, padding, mode=None, value=None))]
    fn pad(
        &mut self,
        py: Python<'_>,
        input: &PyMLOperand,
        padding: Vec<u32>,
        mode: Option<&str>,
        value: Option<Py<PyAny>>,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_pad_shape;

        // Infer output shape
        let output_shape = infer_pad_shape(&input.descriptor.static_or_max_shape(), &padding)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        // Validate mode
        let mode_str = mode.unwrap_or("constant");
        if !["constant", "edge", "reflection", "symmetric"].contains(&mode_str) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid pad mode '{}', must be 'constant', 'edge', 'reflection', or 'symmetric'",
                mode_str
            )));
        }

        let half = padding.len() / 2;
        let beginning_padding = padding[..half].to_vec();
        let ending_padding = padding[half..].to_vec();

        let pad_opts = MLPadOptions {
            label: String::new(),
            mode: mode_str.to_string(),
            value: pad_options_value(py, value)?,
        };

        self.push_op(Operation::Pad {
            input: input.id,
            beginning_padding,
            ending_padding,
            options: Some(pad_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// GELU activation operation
    ///
    /// Args:
    ///     input: The input tensor
    ///
    /// Returns:
    ///     MLOperand: Output operand with GELU activation applied
    fn gelu(&mut self, input: &PyMLOperand) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_gelu_shape;

        let output_shape = infer_gelu_shape(&input.descriptor.static_or_max_shape());

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::Gelu {
            input: input.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Squeeze operation (remove dimensions of size 1)
    ///
    /// Args:
    ///     input: The input tensor
    ///     axes: Optional sequence of axes to squeeze. If None, all dimensions of size 1 are removed
    ///
    /// Returns:
    ///     MLOperand: Output operand with dimensions squeezed
    #[pyo3(signature = (input, axes=None))]
    fn squeeze(&mut self, input: &PyMLOperand, axes: Option<Vec<u32>>) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_squeeze_shape;

        let output_shape =
            infer_squeeze_shape(&input.descriptor.static_or_max_shape(), axes.as_deref())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let squeeze_opts = MLSqueezeOptions {
            label: String::new(),
            axes: axes.unwrap_or_default(),
        };

        self.push_op(Operation::Squeeze {
            input: input.id,
            options: Some(squeeze_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Unsqueeze operation (add dimensions of size 1)
    ///
    /// Args:
    ///     input: The input tensor
    ///     axes: Sequence of axes where dimensions of size 1 should be inserted
    ///
    /// Returns:
    ///     MLOperand: Output operand with dimensions unsqueezed
    fn unsqueeze(&mut self, input: &PyMLOperand, axes: Vec<u32>) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_unsqueeze_shape;

        let output_shape = infer_unsqueeze_shape(&input.descriptor.static_or_max_shape(), &axes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let unsqueeze_opts = MLUnsqueezeOptions {
            label: String::new(),
            axes,
        };

        self.push_op(Operation::Unsqueeze {
            input: input.id,
            options: Some(unsqueeze_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// ArgMax operation (find indices of maximum values)
    ///
    /// Args:
    ///     input: The input tensor
    ///     axis: The axis to reduce along
    ///     keep_dimensions: If True, keep the reduced axis with size 1. Default is False
    ///     output_data_type: Output data type for indices ("int32" or "int64"). Default is "int64"
    ///
    /// Returns:
    ///     MLOperand: Output operand containing indices of maximum values
    #[pyo3(signature = (input, axis, keep_dimensions=false, output_data_type=None))]
    fn arg_max(
        &mut self,
        input: &PyMLOperand,
        axis: u32,
        keep_dimensions: bool,
        output_data_type: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_arg_reduce_shape;

        let output_shape = infer_arg_reduce_shape(
            &input.descriptor.static_or_max_shape(),
            axis,
            keep_dimensions,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Parse output data type, default to int64
        let (output_type, ml_output_dtype) = match output_data_type {
            Some("int32") => (DataType::Int32, MLOperandDataType::Int32),
            Some("int64") | None => (DataType::Int64, MLOperandDataType::Int64),
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid output_data_type '{}', must be 'int32' or 'int64'",
                    other
                )));
            }
        };

        let output_descriptor = OperandDescriptor {
            data_type: output_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let arg_opts = MLArgMinMaxOptions {
            label: String::new(),
            keep_dimensions,
            output_data_type: ml_output_dtype,
        };

        self.push_op(Operation::ArgMax {
            input: input.id,
            axis,
            options: Some(arg_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// ArgMin operation (find indices of minimum values)
    ///
    /// Args:
    ///     input: The input tensor
    ///     axis: The axis to reduce along
    ///     keep_dimensions: If True, keep the reduced axis with size 1. Default is False
    ///     output_data_type: Output data type for indices ("int32" or "int64"). Default is "int64"
    ///
    /// Returns:
    ///     MLOperand: Output operand containing indices of minimum values
    #[pyo3(signature = (input, axis, keep_dimensions=false, output_data_type=None))]
    fn arg_min(
        &mut self,
        input: &PyMLOperand,
        axis: u32,
        keep_dimensions: bool,
        output_data_type: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_arg_reduce_shape;

        let output_shape = infer_arg_reduce_shape(
            &input.descriptor.static_or_max_shape(),
            axis,
            keep_dimensions,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Parse output data type, default to int64
        let (output_type, ml_output_dtype) = match output_data_type {
            Some("int32") => (DataType::Int32, MLOperandDataType::Int32),
            Some("int64") | None => (DataType::Int64, MLOperandDataType::Int64),
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid output_data_type '{}', must be 'int32' or 'int64'",
                    other
                )));
            }
        };

        let output_descriptor = OperandDescriptor {
            data_type: output_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let arg_opts = MLArgMinMaxOptions {
            label: String::new(),
            keep_dimensions,
            output_data_type: ml_output_dtype,
        };

        self.push_op(Operation::ArgMin {
            input: input.id,
            axis,
            options: Some(arg_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Cast operation (type conversion)
    ///
    /// Args:
    ///     input: The input tensor
    ///     data_type: Target data type ("float32", "float16", "int32", "uint32", "int8", "uint8", "int64")
    ///
    /// Returns:
    ///     MLOperand: Output operand with converted type
    fn cast(&mut self, input: &PyMLOperand, data_type: &str) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_cast_shape;

        let output_shape = infer_cast_shape(&input.descriptor.static_or_max_shape());

        // Parse target data type
        let (target_type, ml_dtype) = match data_type {
            "float32" => (DataType::Float32, MLOperandDataType::Float32),
            "float16" => (DataType::Float16, MLOperandDataType::Float16),
            "int32" => (DataType::Int32, MLOperandDataType::Int32),
            "uint32" => (DataType::Uint32, MLOperandDataType::Uint32),
            "int8" => (DataType::Int8, MLOperandDataType::Int8),
            "uint8" => (DataType::Uint8, MLOperandDataType::Uint8),
            "int64" => (DataType::Int64, MLOperandDataType::Int64),
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid data_type '{}', must be one of: float32, float16, int32, uint32, int8, uint8, int64",
                    other
                )));
            }
        };

        let output_descriptor = OperandDescriptor {
            data_type: target_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::Cast {
            input: input.id,
            data_type: ml_dtype,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Build the computational graph
    ///
    /// Args:
    ///     outputs: Dictionary mapping output names to MLOperand objects
    ///
    /// Returns:
    ///     MLGraph: The compiled graph
    fn build(&mut self, py: Python<'_>, outputs: &Bound<'_, PyDict>) -> PyResult<PyMLGraph> {
        // Operation results are tagged Output during construction; demote before binding graph outputs.
        for operand in &mut self.operands {
            if matches!(operand.kind, OperandKind::Output) {
                operand.kind = OperandKind::Intermediate;
                operand.name = None;
            }
        }

        let mut output_operands = Vec::new();

        // Mark outputs and collect output IDs
        for (name, operand_obj) in outputs.iter() {
            let name_str: String = name.extract()?;
            let operand: PyMLOperand = operand_obj.extract()?;

            // Update the operand to mark it as an output with the given name
            if let Some(op) = self.operands.get_mut(operand.id as usize) {
                op.kind = OperandKind::Output;
                op.name = Some(name_str);
            }
            output_operands.push(operand.id);
        }

        // Create GraphInfo
        let graph_info = GraphInfo {
            operands: self.operands.clone(),
            input_operands: self.input_operands.clone(),
            output_operands,
            operations: self.operations.clone(),
            constant_operand_ids_to_handles: self.constant_data_map.clone(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        // Validate the graph
        let validator = GraphValidator::new(&graph_info, Default::default());
        validator.validate().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Graph validation failed: {}", e))
        })?;

        if self.built {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "MLGraphBuilder.build() was already called",
            ));
        }
        self.built = true;

        let graph_slot = {
            self.context
                .bind(py)
                .borrow()
                .compile_graph(graph_info.clone())?
        };

        Ok(PyMLGraph::new_compiled(
            graph_info,
            self.context.clone_ref(py),
            graph_slot,
        ))
    }

    /// Scatter elements operation
    ///
    /// Updates values in input tensor at indices specified by indices tensor.
    ///
    /// Args:
    ///     input: The base tensor to scatter values into
    ///     indices: Integer tensor of same rank as input, containing indices
    ///     updates: Tensor of same rank as input, containing values to scatter
    ///     axis: Axis along which to scatter (can be negative)
    ///
    /// Returns:
    ///     MLOperand: Output operand with scattered values
    fn scatter_elements(
        &mut self,
        input: &PyMLOperand,
        indices: &PyMLOperand,
        updates: &PyMLOperand,
        axis: u32,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_scatter_elements_shape;

        let output_shape = infer_scatter_elements_shape(
            &input.descriptor.static_or_max_shape(),
            &indices.descriptor.static_or_max_shape(),
            &updates.descriptor.static_or_max_shape(),
            axis,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let scatter_opts = MLScatterOptions {
            label: String::new(),
            axis,
        };

        self.push_op(Operation::ScatterElements {
            input: input.id,
            indices: indices.id,
            updates: updates.id,
            options: Some(scatter_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// ScatterND operation
    ///
    /// Scatter updates into a tensor using multi-dimensional indices.
    ///
    /// Args:
    ///     input: Base tensor of rank r >= 1
    ///     indices: Integer tensor of rank q >= 1
    ///     updates: Tensor containing values to scatter
    ///
    /// Returns:
    ///     MLOperand: Output operand with scattered values
    fn scatter_nd(
        &mut self,
        input: &PyMLOperand,
        indices: &PyMLOperand,
        updates: &PyMLOperand,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_scatter_nd_shape;

        let output_shape = infer_scatter_nd_shape(
            &input.descriptor.static_or_max_shape(),
            &indices.descriptor.static_or_max_shape(),
            &updates.descriptor.static_or_max_shape(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::ScatterND {
            input: input.id,
            indices: indices.id,
            updates: updates.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Tile operation
    ///
    /// Repeats tensor along each dimension according to repetitions.
    ///
    /// Args:
    ///     input: Input tensor to tile
    ///     repetitions: Number of repetitions for each dimension
    ///
    /// Returns:
    ///     MLOperand: Output operand with tiled values
    fn tile(&mut self, input: &PyMLOperand, repetitions: Vec<u32>) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_tile_shape;

        let output_shape = infer_tile_shape(&input.descriptor.static_or_max_shape(), &repetitions)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::Tile {
            input: input.id,
            repetitions,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Triangular operation
    ///
    /// Extract upper or lower triangular part of matrix (last 2 dimensions).
    ///
    /// Args:
    ///     input: Input tensor (rank >= 2)
    ///     upper: Extract upper triangle if true, lower if false
    ///     diagonal: Diagonal offset (0=main, positive=above, negative=below)
    ///
    /// Returns:
    ///     MLOperand: Output operand with non-triangular elements zeroed
    fn triangular(
        &mut self,
        input: &PyMLOperand,
        upper: bool,
        diagonal: i32,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_triangular_shape;

        let output_shape = infer_triangular_shape(&input.descriptor.static_or_max_shape())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let tri_opts = MLTriangularOptions {
            label: String::new(),
            upper: Some(upper),
            diagonal,
        };

        self.push_op(Operation::Triangular {
            input: input.id,
            options: Some(tri_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// hardSigmoid activation operation
    ///
    /// Computes element-wise: y = max(0, min(1, alpha * x + beta))
    ///
    /// Args:
    ///     input: Input tensor
    ///     alpha: Multiplicative coefficient (default: 0.2)
    ///     beta: Additive offset (default: 0.5)
    ///
    /// Returns:
    ///     MLOperand: Output operand with values clipped to [0, 1]
    #[pyo3(signature = (input, alpha=0.2, beta=0.5))]
    fn hard_sigmoid(
        &mut self,
        input: &PyMLOperand,
        alpha: f32,
        beta: f32,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_hardsigmoid_shape;

        let output_shape = infer_hardsigmoid_shape(&input.descriptor.static_or_max_shape())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let hs_opts = MLHardSigmoidOptions {
            label: String::new(),
            alpha: alpha as f64,
            beta: beta as f64,
        };

        self.push_op(Operation::HardSigmoid {
            input: input.id,
            options: Some(hs_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// hardSwish activation operation
    ///
    /// Computes element-wise: y = x * max(0, min(1, alpha * x + beta))
    ///
    /// Args:
    ///     input: Input tensor
    ///     alpha: Multiplicative coefficient for hardSigmoid (default: 1/6)
    ///     beta: Additive offset for hardSigmoid (default: 0.5)
    ///
    /// Returns:
    ///     MLOperand: Output operand
    #[pyo3(signature = (input, alpha=0.166_666_67, beta=0.5))]
    fn hard_swish(&mut self, input: &PyMLOperand, alpha: f32, beta: f32) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_hardswish_shape;

        let _ = (alpha, beta);

        let output_shape = infer_hardswish_shape(&input.descriptor.static_or_max_shape())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::HardSwish {
            input: input.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// softplus activation operation
    ///
    /// Computes element-wise: y = log(1 + exp(x))
    /// Smooth approximation of ReLU
    ///
    /// Args:
    ///     input: Input tensor
    ///
    /// Returns:
    ///     MLOperand: Output operand with positive values
    fn softplus(&mut self, input: &PyMLOperand) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_softplus_shape;

        let output_shape = infer_softplus_shape(&input.descriptor.static_or_max_shape())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::Softplus {
            input: input.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// softsign activation operation
    ///
    /// Computes element-wise: y = x / (1 + |x|)
    /// Bounded activation with output in (-1, 1)
    ///
    /// Args:
    ///     input: Input tensor
    ///
    /// Returns:
    ///     MLOperand: Output operand with values in (-1, 1)
    fn softsign(&mut self, input: &PyMLOperand) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_softsign_shape;

        let output_shape = infer_softsign_shape(&input.descriptor.static_or_max_shape())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::Softsign {
            input: input.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// clamp operation (element-wise clamping)
    ///
    /// Constrains every element in the input tensor between min and max values:
    ///   y = max(min_value, min(x, max_value))
    ///
    /// Args:
    ///     input: Input tensor
    ///     min_value: Minimum value (default: negative infinity)
    ///     max_value: Maximum value (default: positive infinity)
    ///
    /// Returns:
    ///     MLOperand: Output operand with values clamped to [min_value, max_value]
    ///
    /// Example:
    ///     # ReLU6: clamp(x, 0, 6)
    ///     relu6 = builder.clamp(x, min_value=0.0, max_value=6.0)
    #[pyo3(signature = (input, min_value=None, max_value=None))]
    fn clamp(
        &mut self,
        py: Python<'_>,
        input: &PyMLOperand,
        min_value: Option<Py<PyAny>>,
        max_value: Option<Py<PyAny>>,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_clamp_shape;

        let (min_json, max_json) = clamp_limits_ordered(py, min_value, max_value)?;

        let output_shape = infer_clamp_shape(&input.descriptor.static_or_max_shape());

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let clamp_opts = MLClampOptions {
            label: String::new(),
            min_value: Some(min_json),
            max_value: Some(max_json),
        };

        self.push_op(Operation::Clamp {
            input: input.id,
            options: Some(clamp_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// elu activation operation (Exponential Linear Unit)
    ///
    /// Computes element-wise:
    ///   y = x if x >= 0
    ///   y = alpha * (exp(x) - 1) if x < 0
    ///
    /// Args:
    ///     input: Input tensor
    ///     alpha: Coefficient for negative values (default: 1.0)
    ///
    /// Returns:
    ///     MLOperand: Output operand
    #[pyo3(signature = (input, alpha=1.0))]
    fn elu(&mut self, input: &PyMLOperand, alpha: f32) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_elu_shape;

        let output_shape = infer_elu_shape(&input.descriptor.static_or_max_shape())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let elu_opts = MLEluOptions {
            label: String::new(),
            alpha: alpha as f64,
        };

        self.push_op(Operation::Elu {
            input: input.id,
            options: Some(elu_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// leakyRelu activation operation (Leaky Rectified Linear Unit)
    ///
    /// Computes element-wise:
    ///   y = x if x >= 0
    ///   y = alpha * x if x < 0
    ///
    /// Args:
    ///     input: Input tensor
    ///     alpha: Leakage coefficient for negative values (default: 0.01)
    ///
    /// Returns:
    ///     MLOperand: Output operand
    #[pyo3(signature = (input, alpha=0.01))]
    fn leaky_relu(&mut self, input: &PyMLOperand, alpha: f32) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_leakyrelu_shape;

        let output_shape = infer_leakyrelu_shape(&input.descriptor.static_or_max_shape())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let leaky_opts = MLLeakyReluOptions {
            label: String::new(),
            alpha: alpha as f64,
        };

        self.push_op(Operation::LeakyRelu {
            input: input.id,
            options: Some(leaky_opts),
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// prelu activation operation (Parametric Rectified Linear Unit)
    ///
    /// Computes element-wise:
    ///   y = x if x >= 0
    ///   y = slope * x if x < 0
    ///
    /// Args:
    ///     input: Input tensor
    ///     slope: Learnable slope tensor (must be unidirectionally broadcastable to input)
    ///
    /// Returns:
    ///     MLOperand: Output operand
    fn prelu(&mut self, input: &PyMLOperand, slope: &PyMLOperand) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::infer_prelu_shape;

        let output_shape = infer_prelu_shape(
            &input.descriptor.static_or_max_shape(),
            &slope.descriptor.static_or_max_shape(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::Prelu {
            input: input.id,
            slope: slope.id,
            options: None,
            outputs: vec![output_id],
        });

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Linear activation: output = alpha * input + beta
    #[pyo3(signature = (input, alpha=1.0, beta=0.0))]
    fn linear(&mut self, input: &PyMLOperand, alpha: f32, beta: f32) -> PyResult<PyMLOperand> {
        let output_descriptor = input.descriptor.clone();
        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let linear_opts = MLLinearOptions {
            label: String::new(),
            alpha: alpha as f64,
            beta: beta as f64,
        };

        self.push_op(Operation::Linear {
            input: input.id,
            options: Some(linear_opts),
            outputs: vec![output_id],
        });

        self.register_output_operand(output_id, output_descriptor)
    }

    /// Element-wise is-NaN test (uint8 output)
    fn is_nan(&mut self, input: &PyMLOperand) -> PyResult<PyMLOperand> {
        let mut output_descriptor = input.descriptor.clone();
        output_descriptor.data_type = DataType::Uint8;
        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::IsNaN {
            input: input.id,
            options: None,
            outputs: vec![output_id],
        });

        self.register_output_operand(output_id, output_descriptor)
    }

    /// Element-wise is-infinite test (uint8 output)
    fn is_infinite(&mut self, input: &PyMLOperand) -> PyResult<PyMLOperand> {
        let mut output_descriptor = input.descriptor.clone();
        output_descriptor.data_type = DataType::Uint8;
        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::IsInfinite {
            input: input.id,
            options: None,
            outputs: vec![output_id],
        });

        self.register_output_operand(output_id, output_descriptor)
    }

    /// Returns the rank of the input as a 1D int64 tensor
    fn shape(&mut self, input: &PyMLOperand) -> PyResult<PyMLOperand> {
        let rank = input.descriptor.static_or_max_shape().len() as u32;
        let output_descriptor = OperandDescriptor {
            data_type: DataType::Int64,
            shape: to_dimension_vector(&[rank]),
            pending_permutation: Vec::new(),
        };
        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::Shape {
            input: input.id,
            options: None,
            outputs: vec![output_id],
        });

        self.register_output_operand(output_id, output_descriptor)
    }

    /// Cumulative sum along an axis
    #[pyo3(signature = (input, axis, exclusive=false, reversed=false))]
    fn cumulative_sum(
        &mut self,
        input: &PyMLOperand,
        axis: u32,
        exclusive: bool,
        reversed: bool,
    ) -> PyResult<PyMLOperand> {
        let output_descriptor = input.descriptor.clone();
        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let opts = MLCumulativeSumOptions {
            label: String::new(),
            exclusive,
            reversed,
        };

        self.push_op(Operation::CumulativeSum {
            input: input.id,
            axis,
            options: Some(opts),
            outputs: vec![output_id],
        });

        self.register_output_operand(output_id, output_descriptor)
    }

    /// Reverse tensor along axes (default: all axes)
    #[pyo3(signature = (input, axes=None))]
    fn reverse(&mut self, input: &PyMLOperand, axes: Option<Vec<u32>>) -> PyResult<PyMLOperand> {
        let output_descriptor = input.descriptor.clone();
        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let reverse_opts = MLReverseOptions {
            label: String::new(),
            axes,
        };

        self.push_op(Operation::Reverse {
            input: input.id,
            options: Some(reverse_opts),
            outputs: vec![output_id],
        });

        self.register_output_operand(output_id, output_descriptor)
    }

    /// 2D resampling (resize spatial axes)
    #[pyo3(signature = (input, sizes=None, scales=None, mode="nearest-neighbor", axes=None))]
    fn resample2d(
        &mut self,
        input: &PyMLOperand,
        sizes: Option<Vec<u32>>,
        scales: Option<Vec<f32>>,
        mode: &str,
        axes: Option<Vec<u32>>,
    ) -> PyResult<PyMLOperand> {
        let input_shape = input.descriptor.shape.clone();
        let rank = input_shape.len();
        let axis_pair: [usize; 2] = if let Some(ref a) = axes {
            if a.len() != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "resample2d axes must have length 2",
                ));
            }
            [a[0] as usize, a[1] as usize]
        } else {
            [rank.saturating_sub(2), rank.saturating_sub(1)]
        };

        let resample_opts = MLResample2dOptions {
            label: String::new(),
            mode: mode.to_string(),
            scales: scales.unwrap_or_default(),
            sizes,
            axes: axes.unwrap_or_else(|| vec![axis_pair[0] as u32, axis_pair[1] as u32]),
        };

        let output_shape = infer_resample2d_shape(
            &input_shape,
            axis_pair,
            resample_opts.sizes.as_deref(),
            &resample_opts.scales,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };
        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        self.push_op(Operation::Resample2d {
            input: input.id,
            options: Some(resample_opts),
            outputs: vec![output_id],
        });

        self.register_output_operand(output_id, output_descriptor)
    }

    /// GRU recurrent network
    #[pyo3(signature = (input, weight, recurrent_weight, steps, hidden_size, bias=None, recurrent_bias=None, initial_hidden_state=None, reset_after=false, return_sequence=false, direction="forward", layout="zrn", activations=None))]
    fn gru(
        &mut self,
        input: &PyMLOperand,
        weight: &PyMLOperand,
        recurrent_weight: &PyMLOperand,
        steps: u32,
        hidden_size: u32,
        bias: Option<&PyMLOperand>,
        recurrent_bias: Option<&PyMLOperand>,
        initial_hidden_state: Option<&PyMLOperand>,
        reset_after: bool,
        return_sequence: bool,
        direction: &str,
        layout: &str,
        activations: Option<Vec<String>>,
    ) -> PyResult<Vec<PyMLOperand>> {
        let num_dir = recurrent_num_directions(direction);
        let batch = recurrent_batch_size(&input.descriptor.static_or_max_shape());
        let dtype = input.descriptor.data_type;

        let mut output_shapes = vec![vec![num_dir, batch, hidden_size]];
        if return_sequence {
            output_shapes.push(vec![steps, num_dir, batch, hidden_size]);
        }

        let output_ids: Vec<u32> = (0..output_shapes.len() as u32)
            .map(|i| self.next_operand_id + i)
            .collect();
        self.next_operand_id += output_shapes.len() as u32;

        let options = MLGruOptions {
            label: String::new(),
            bias: bias.map(|o| o.id),
            recurrent_bias: recurrent_bias.map(|o| o.id),
            initial_hidden_state: initial_hidden_state.map(|o| o.id),
            reset_after,
            return_sequence,
            direction: direction.to_string(),
            layout: layout.to_string(),
            activations,
        };

        self.push_op(Operation::Gru {
            input: input.id,
            weight: weight.id,
            recurrence: recurrent_weight.id,
            steps,
            hidden_size,
            options: Some(options),
            outputs: output_ids.clone(),
        });

        self.register_multi_output_operands(&output_ids, dtype, &output_shapes)
    }

    /// GRU cell (single step)
    #[pyo3(signature = (input, weight, recurrent_weight, hidden_state, hidden_size, bias=None, recurrent_bias=None, reset_after=false, layout="zrn", activations=None))]
    fn gru_cell(
        &mut self,
        input: &PyMLOperand,
        weight: &PyMLOperand,
        recurrent_weight: &PyMLOperand,
        hidden_state: &PyMLOperand,
        hidden_size: u32,
        bias: Option<&PyMLOperand>,
        recurrent_bias: Option<&PyMLOperand>,
        reset_after: bool,
        layout: &str,
        activations: Option<Vec<String>>,
    ) -> PyResult<PyMLOperand> {
        let output_descriptor = if hidden_state.descriptor.static_or_max_shape().is_empty() {
            let input_shape = input.descriptor.static_or_max_shape();
            OperandDescriptor {
                data_type: input.descriptor.data_type,
                shape: to_dimension_vector(&[input_shape[0], hidden_size]),
                pending_permutation: Vec::new(),
            }
        } else {
            hidden_state.descriptor.clone()
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let options = MLGruCellOptions {
            label: String::new(),
            bias: bias.map(|o| o.id),
            recurrent_bias: recurrent_bias.map(|o| o.id),
            reset_after,
            layout: layout.to_string(),
            activations,
        };

        self.push_op(Operation::GruCell {
            input: input.id,
            weight: weight.id,
            recurrence: recurrent_weight.id,
            hidden_state: hidden_state.id,
            hidden_size,
            options: Some(options),
            outputs: vec![output_id],
        });

        self.register_output_operand(output_id, output_descriptor)
    }

    /// LSTM recurrent network
    #[pyo3(signature = (input, weight, recurrent_weight, steps, hidden_size, bias=None, recurrent_bias=None, peephole_weight=None, initial_hidden_state=None, initial_cell_state=None, return_sequence=false, direction="forward", layout="iofg", activations=None))]
    fn lstm(
        &mut self,
        input: &PyMLOperand,
        weight: &PyMLOperand,
        recurrent_weight: &PyMLOperand,
        steps: u32,
        hidden_size: u32,
        bias: Option<&PyMLOperand>,
        recurrent_bias: Option<&PyMLOperand>,
        peephole_weight: Option<&PyMLOperand>,
        initial_hidden_state: Option<&PyMLOperand>,
        initial_cell_state: Option<&PyMLOperand>,
        return_sequence: bool,
        direction: &str,
        layout: &str,
        activations: Option<Vec<String>>,
    ) -> PyResult<Vec<PyMLOperand>> {
        let num_dir = recurrent_num_directions(direction);
        let batch = recurrent_batch_size(&input.descriptor.static_or_max_shape());
        let dtype = input.descriptor.data_type;

        let mut output_shapes = vec![
            vec![num_dir, batch, hidden_size],
            vec![num_dir, batch, hidden_size],
        ];
        if return_sequence {
            output_shapes.push(vec![steps, num_dir, batch, hidden_size]);
        }

        let output_ids: Vec<u32> = (0..output_shapes.len() as u32)
            .map(|i| self.next_operand_id + i)
            .collect();
        self.next_operand_id += output_shapes.len() as u32;

        let options = MLLstmOptions {
            label: String::new(),
            bias: bias.map(|o| o.id),
            recurrent_bias: recurrent_bias.map(|o| o.id),
            peephole_weight: peephole_weight.map(|o| o.id),
            initial_hidden_state: initial_hidden_state.map(|o| o.id),
            initial_cell_state: initial_cell_state.map(|o| o.id),
            return_sequence,
            direction: direction.to_string(),
            layout: layout.to_string(),
            activations,
        };

        self.push_op(Operation::Lstm {
            input: input.id,
            weight: weight.id,
            recurrence: recurrent_weight.id,
            steps,
            hidden_size,
            options: Some(options),
            outputs: output_ids.clone(),
        });

        self.register_multi_output_operands(&output_ids, dtype, &output_shapes)
    }

    /// LSTM cell (single step)
    #[pyo3(signature = (input, weight, recurrent_weight, hidden_state, cell_state, hidden_size, bias=None, recurrent_bias=None, peephole_weight=None, layout="iofg", activations=None))]
    fn lstm_cell(
        &mut self,
        input: &PyMLOperand,
        weight: &PyMLOperand,
        recurrent_weight: &PyMLOperand,
        hidden_state: &PyMLOperand,
        cell_state: &PyMLOperand,
        hidden_size: u32,
        bias: Option<&PyMLOperand>,
        recurrent_bias: Option<&PyMLOperand>,
        peephole_weight: Option<&PyMLOperand>,
        layout: &str,
        activations: Option<Vec<String>>,
    ) -> PyResult<Vec<PyMLOperand>> {
        let output_ids = vec![self.next_operand_id, self.next_operand_id + 1];
        self.next_operand_id += 2;

        let options = MLLstmCellOptions {
            label: String::new(),
            bias: bias.map(|o| o.id),
            recurrent_bias: recurrent_bias.map(|o| o.id),
            peephole_weight: peephole_weight.map(|o| o.id),
            layout: layout.to_string(),
            activations,
        };

        self.push_op(Operation::LstmCell {
            input: input.id,
            weight: weight.id,
            recurrence: recurrent_weight.id,
            hidden_state: hidden_state.id,
            cell_state: cell_state.id,
            hidden_size,
            options: Some(options),
            outputs: output_ids.clone(),
        });

        let shapes = [
            hidden_state.descriptor.clone(),
            cell_state.descriptor.clone(),
        ];
        let mut py_operands = Vec::with_capacity(2);
        for (id, descriptor) in output_ids.into_iter().zip(shapes) {
            py_operands.push(self.register_output_operand(id, descriptor)?);
        }
        Ok(py_operands)
    }

    /// Query the static/max shape of an operand without building the graph
    fn operand_shape(&self, operand: &PyMLOperand) -> Vec<u32> {
        operand.descriptor.static_or_max_shape()
    }

    /// Query the data type of an operand
    fn operand_data_type(&self, operand: &PyMLOperand) -> String {
        match operand.descriptor.data_type {
            DataType::Int4 => "int4".to_string(),
            DataType::Uint4 => "uint4".to_string(),
            DataType::Float32 => "float32".to_string(),
            DataType::Float16 => "float16".to_string(),
            DataType::Int32 => "int32".to_string(),
            DataType::Uint32 => "uint32".to_string(),
            DataType::Int8 => "int8".to_string(),
            DataType::Uint8 => "uint8".to_string(),
            DataType::Int64 => "int64".to_string(),
            DataType::Uint64 => "uint64".to_string(),
        }
    }
}

impl PyMLGraphBuilder {
    #[inline]
    pub(crate) fn push_op(&mut self, op: Operation) {
        self.operations.push(op);
    }

    fn register_output_operand(
        &mut self,
        output_id: u32,
        output_descriptor: OperandDescriptor,
    ) -> PyResult<PyMLOperand> {
        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());
        Ok(py_operand)
    }

    fn register_multi_output_operands(
        &mut self,
        output_ids: &[u32],
        data_type: DataType,
        output_shapes: &[Vec<u32>],
    ) -> PyResult<Vec<PyMLOperand>> {
        let mut py_operands = Vec::with_capacity(output_ids.len());
        for (id, shape) in output_ids.iter().zip(output_shapes.iter()) {
            let descriptor = OperandDescriptor {
                data_type,
                shape: to_dimension_vector(shape),
                pending_permutation: Vec::new(),
            };
            py_operands.push(self.register_output_operand(*id, descriptor)?);
        }
        Ok(py_operands)
    }

    /// Create a new graph builder tied to a context (internal).
    pub(crate) fn new_for_context(context: Py<super::context::PyMLContext>) -> Self {
        Self {
            context,
            built: false,
            operands: Vec::new(),
            operations: Vec::new(),
            input_operands: Vec::new(),
            next_operand_id: 0,
            operand_map: HashMap::new(),
            constant_data_map: HashMap::new(),
        }
    }

    /// Helper for binary operations with broadcasting
    fn binary_op(
        &mut self,
        op_type: &str,
        a: &PyMLOperand,
        b: &PyMLOperand,
    ) -> PyResult<PyMLOperand> {
        // Compute broadcasted output shape
        let output_shape = broadcast_shapes(
            &a.descriptor.static_or_max_shape(),
            &b.descriptor.static_or_max_shape(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: a.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let op = match op_type {
            "add" => Operation::Add {
                a: a.id,
                b: b.id,
                options: None,
                outputs: vec![output_id],
            },
            "sub" => Operation::Sub {
                a: a.id,
                b: b.id,
                options: None,
                outputs: vec![output_id],
            },
            "mul" => Operation::Mul {
                a: a.id,
                b: b.id,
                options: None,
                outputs: vec![output_id],
            },
            "div" => Operation::Div {
                a: a.id,
                b: b.id,
                options: None,
                outputs: vec![output_id],
            },
            "pow" => Operation::Pow {
                a: a.id,
                b: b.id,
                options: None,
                outputs: vec![output_id],
            },
            "max" => Operation::Max {
                a: a.id,
                b: b.id,
                options: None,
                outputs: vec![output_id],
            },
            "min" => Operation::Min {
                a: a.id,
                b: b.id,
                options: None,
                outputs: vec![output_id],
            },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "internal error: unsupported binary op '{}'",
                    op_type
                )));
            }
        };
        self.push_op(op);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Helper for unary operations
    fn unary_op(&mut self, op_type: &str, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        let output_descriptor = x.descriptor.clone();

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let input = x.id;
        let outputs = vec![output_id];
        let op = match op_type {
            "abs" => Operation::Abs {
                input,
                options: None,
                outputs,
            },
            "ceil" => Operation::Ceil {
                input,
                options: None,
                outputs,
            },
            "cos" => Operation::Cos {
                input,
                options: None,
                outputs,
            },
            "exp" => Operation::Exp {
                input,
                options: None,
                outputs,
            },
            "floor" => Operation::Floor {
                input,
                options: None,
                outputs,
            },
            "log" => Operation::Log {
                input,
                options: None,
                outputs,
            },
            "neg" => Operation::Neg {
                input,
                options: None,
                outputs,
            },
            "relu" => Operation::Relu {
                input,
                options: None,
                outputs,
            },
            "sigmoid" => Operation::Sigmoid {
                input,
                options: None,
                outputs,
            },
            "sin" => Operation::Sin {
                input,
                options: None,
                outputs,
            },
            "sqrt" => Operation::Sqrt {
                input,
                options: None,
                outputs,
            },
            "tan" => Operation::Tan {
                input,
                options: None,
                outputs,
            },
            "tanh" => Operation::Tanh {
                input,
                options: None,
                outputs,
            },
            "erf" => Operation::Erf {
                input,
                options: None,
                outputs,
            },
            "identity" => Operation::Identity {
                input,
                options: None,
                outputs,
            },
            "reciprocal" => Operation::Reciprocal {
                input,
                options: None,
                outputs,
            },
            "sign" => Operation::Sign {
                input,
                options: None,
                outputs,
            },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unary op '{}' has no Operation variant in this rustnn build",
                    op_type
                )));
            }
        };
        self.push_op(op);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Helper for reduction operations
    fn reduce_op(
        &mut self,
        op_type: &str,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        use rustnn::shape_inference::{infer_reduce_shape, ReduceOptions};

        let infer_axes = match &axes {
            None => (0..input.descriptor.static_or_max_shape().len() as u32).collect(),
            Some(v) => v.clone(),
        };

        let infer_opts = ReduceOptions {
            axes: infer_axes,
            keep_dimensions,
        };

        // Infer output shape
        let output_shape = infer_reduce_shape(&input.descriptor.static_or_max_shape(), &infer_opts)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: to_dimension_vector(&output_shape),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let reduce_opts = MLReduceOptions {
            label: String::new(),
            axes,
            keep_dimensions,
        };

        let op = match op_type {
            "reduceSum" => Operation::ReduceSum {
                input: input.id,
                options: Some(reduce_opts),
                outputs: vec![output_id],
            },
            "reduceMean" => Operation::ReduceMean {
                input: input.id,
                options: Some(reduce_opts),
                outputs: vec![output_id],
            },
            "reduceMax" => Operation::ReduceMax {
                input: input.id,
                options: Some(reduce_opts),
                outputs: vec![output_id],
            },
            "reduceMin" => Operation::ReduceMin {
                input: input.id,
                options: Some(reduce_opts),
                outputs: vec![output_id],
            },
            "reduceProduct" => Operation::ReduceProduct {
                input: input.id,
                options: Some(reduce_opts),
                outputs: vec![output_id],
            },
            "reduceL1" => Operation::ReduceL1 {
                input: input.id,
                options: Some(reduce_opts),
                outputs: vec![output_id],
            },
            "reduceL2" => Operation::ReduceL2 {
                input: input.id,
                options: Some(reduce_opts),
                outputs: vec![output_id],
            },
            "reduceLogSum" => Operation::ReduceLogSum {
                input: input.id,
                options: Some(reduce_opts),
                outputs: vec![output_id],
            },
            "reduceLogSumExp" => Operation::ReduceLogSumExp {
                input: input.id,
                options: Some(reduce_opts),
                outputs: vec![output_id],
            },
            "reduceSumSquare" => Operation::ReduceSumSquare {
                input: input.id,
                options: Some(reduce_opts),
                outputs: vec![output_id],
            },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "internal error: unsupported reduce op '{}'",
                    op_type
                )));
            }
        };
        self.push_op(op);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }
}
