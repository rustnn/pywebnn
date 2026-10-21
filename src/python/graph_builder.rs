//! Graph builder for constructing WebNN computational graphs
//!
//! PyO3 macros generate unsafe code that triggers unsafe_op_in_unsafe_fn warnings.
//! This is expected behavior from the macro-generated code.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::too_many_arguments)]

use super::context_state::{data_type_to_ml_operand, map_rustnn_error, pack_numpy_to_4bit_bytes};
use super::graph::PyMLGraph;
use super::operand::{parse_data_type, PyMLOperand};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustnn::graph::DataType;
use rustnn::mlcontext::{MLGraphBuilder, MLOperand, MLOperandDescriptor};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::{
    MLArgMinMaxOptions, MLBatchNormalizationOptions, MLClampOptions, MLConv2dOptions,
    MLConvTranspose2dOptions, MLCumulativeSumOptions, MLDimension, MLDynamicDimension,
    MLEluOptions, MLGatherOptions, MLGemmOptions, MLGruCellOptions, MLGruOptions,
    MLHardSigmoidOptions, MLInstanceNormalizationOptions, MLLayerNormalizationOptions,
    MLLeakyReluOptions, MLLinearOptions, MLLstmCellOptions, MLLstmOptions, MLOperatorOptions,
    MLPadOptions, MLPool2dOptions, MLReduceOptions, MLResample2dOptions, MLReverseOptions,
    MLScatterOptions, MLSliceOptions, MLSplitOptions, MLSqueezeOptions, MLTransposeOptions,
    MLTriangularOptions, MLUnsqueezeOptions,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

type RuntimeBuilder = MLGraphBuilder<'static, 'static>;

fn parse_dimensions(
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<rustnn::operator_options::MLDimension>> {
    values
        .try_iter()?
        .map(|value| {
            let value = value?;
            if let Ok(size) = value.extract::<u32>() {
                return Ok(MLDimension::Static(size));
            }
            let dimension = value.cast::<PyDict>()?;
            let name = dimension
                .get_item("name")?
                .ok_or_else(|| PyValueError::new_err("Dynamic dimension requires name"))?
                .extract()?;
            let max_size = dimension
                .get_item("maxSize")?
                .ok_or_else(|| PyValueError::new_err("Dynamic dimension requires maxSize"))?
                .extract()?;
            Ok(MLDimension::Dynamic(MLDynamicDimension { name, max_size }))
        })
        .collect()
}

fn builder_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
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
) -> PyResult<Option<serde_json::Value>> {
    match value {
        None => Ok(None),
        Some(obj) => {
            let v = obj.bind(py);
            if let Ok(i) = v.extract::<i64>() {
                return Ok(Some(serde_json::Value::from(i)));
            }
            if let Ok(u) = v.extract::<u64>() {
                return Ok(Some(serde_json::Value::from(u)));
            }
            if let Ok(f) = v.extract::<f64>() {
                return Ok(Some(if f.is_finite() {
                    serde_json::Value::from(f)
                } else {
                    serde_json::Value::String(f.to_string())
                }));
            }
            Err(pyo3::exceptions::PyTypeError::new_err(
                "clamp limit must be a number",
            ))
        }
    }
}

#[pyclass(name = "MLGraphBuilder", unsendable)]
pub struct PyMLGraphBuilder {
    // Drop the backend builder before releasing its owning Python context.
    inner: Option<RuntimeBuilder>,
    context: Py<super::context::PyMLContext>,
    builder_id: u64,
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
        self.inner_mut()?;
        let descriptor = MLOperandDescriptor::new(
            data_type_to_ml_operand(parse_data_type(data_type)?)?,
            shape.into_iter().map(u64::from).collect(),
        );
        let handle = self
            .inner_mut()?
            .input(&name, &descriptor)
            .map_err(map_rustnn_error)?;
        self.wrap(handle)
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
        self.inner_mut()?;

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

        let cast_options = PyDict::new(py);
        cast_options.set_item("copy", false)?;
        let array = match actual_dtype {
            DataType::Int4 | DataType::Uint4 => array,
            _ => array.call_method(
                "astype",
                (data_type_to_ml_operand(actual_dtype)?.as_str(),),
                Some(&cast_options),
            )?,
        };

        // Convert array to bytes (nibble-packed for int4/uint4)
        let bytes = match actual_dtype {
            DataType::Int4 | DataType::Uint4 => pack_numpy_to_4bit_bytes(array, actual_dtype)?,
            _ => array.call_method0("tobytes")?.extract()?,
        };

        let descriptor = MLOperandDescriptor::new(
            data_type_to_ml_operand(actual_dtype)?,
            actual_shape.into_iter().map(u64::from).collect(),
        );
        let handle = self
            .inner_mut()?
            .constant_from_bytes(&descriptor, bytes)
            .map_err(map_rustnn_error)?;
        self.wrap(handle)
    }

    // Binary operations

    /// Element-wise addition
    #[pyo3(signature = (a, b, *, label=None))]
    fn add(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let a = self.handle(a)?;
        let b = self.handle(b)?;

        self.call(|builder| {
            builder.add_with_options(
                a,
                b,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise subtraction
    #[pyo3(signature = (a, b, *, label=None))]
    fn sub(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let a = self.handle(a)?;
        let b = self.handle(b)?;

        self.call(|builder| {
            builder.sub_with_options(
                a,
                b,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise multiplication
    #[pyo3(signature = (a, b, *, label=None))]
    fn mul(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let a = self.handle(a)?;
        let b = self.handle(b)?;

        self.call(|builder| {
            builder.mul_with_options(
                a,
                b,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise division
    #[pyo3(signature = (a, b, *, label=None))]
    fn div(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let a = self.handle(a)?;
        let b = self.handle(b)?;

        self.call(|builder| {
            builder.div_with_options(
                a,
                b,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise power
    #[pyo3(signature = (a, b, *, label=None))]
    fn pow(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let a = self.handle(a)?;
        let b = self.handle(b)?;

        self.call(|builder| {
            builder.pow_with_options(
                a,
                b,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise maximum
    #[pyo3(signature = (a, b, *, label=None))]
    fn max(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let a = self.handle(a)?;
        let b = self.handle(b)?;

        self.call(|builder| {
            builder.max_with_options(
                a,
                b,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise minimum
    #[pyo3(signature = (a, b, *, label=None))]
    fn min(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let a = self.handle(a)?;
        let b = self.handle(b)?;

        self.call(|builder| {
            builder.min_with_options(
                a,
                b,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Matrix multiplication
    #[pyo3(signature = (a, b, *, label=None))]
    fn matmul(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let a = self.handle(a)?;
        let b = self.handle(b)?;

        self.call(|builder| {
            builder.matmul_with_options(
                a,
                b,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
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
    #[pyo3(signature = (a, b, c=None, alpha=1.0, beta=1.0, a_transpose=false, b_transpose=false, *, label=None))]
    fn gemm(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        c: Option<&PyMLOperand>,
        alpha: f64,
        beta: f64,
        a_transpose: bool,
        b_transpose: bool,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let a = self.handle(a)?;
        let b = self.handle(b)?;
        let c = c.map(|operand| self.handle(operand)).transpose()?;
        let options = MLGemmOptions {
            label: label.unwrap_or_default().to_string(),
            c: c.map(MLOperand::rustnn_index),
            alpha,
            beta,
            a_transpose,
            b_transpose,
        };
        self.call(|builder| builder.gemm_with_options(a, b, options))
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
    #[pyo3(signature = (input, filter, strides=None, dilations=None, pads=None, groups=None, input_layout=None, filter_layout=None, bias=None, *, label=None))]
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
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let filter = self.handle(filter)?;
        let bias = bias.map(|operand| self.handle(operand)).transpose()?;
        let options = MLConv2dOptions {
            label: label.unwrap_or_default().to_string(),
            padding: pads.unwrap_or_else(|| vec![0; 4]),
            strides: strides.unwrap_or_else(|| vec![1; 2]),
            dilations: dilations.unwrap_or_else(|| vec![1; 2]),
            groups: groups.unwrap_or(1),
            input_layout: input_layout.unwrap_or("nchw").parse().map_err(|_| {
                PyValueError::new_err("Invalid input_layout: expected 'nchw' or 'nhwc'")
            })?,
            filter_layout: filter_layout.unwrap_or("oihw").parse().map_err(|_| {
                PyValueError::new_err(
                    "Invalid filter_layout: expected 'oihw', 'hwio', 'ohwi', or 'ihwo'",
                )
            })?,
            bias: bias.map(MLOperand::rustnn_index),
        };
        self.call(|builder| builder.conv2d_with_options(input, filter, options))
    }

    /// 2D Transposed Convolution operation (deconvolution)
    #[pyo3(signature = (input, filter, strides=None, dilations=None, pads=None, output_padding=None, output_sizes=None, groups=None, input_layout=None, filter_layout=None, bias=None, *, label=None))]
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
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let filter = self.handle(filter)?;
        let bias = bias.map(|operand| self.handle(operand)).transpose()?;
        let options = MLConvTranspose2dOptions {
            label: label.unwrap_or_default().to_string(),
            padding: pads.unwrap_or_else(|| vec![0; 4]),
            strides: strides.unwrap_or_else(|| vec![1; 2]),
            dilations: dilations.unwrap_or_else(|| vec![1; 2]),
            groups: groups.unwrap_or(1),
            input_layout: input_layout.unwrap_or("nchw").parse().map_err(|_| {
                PyValueError::new_err("Invalid input_layout: expected 'nchw' or 'nhwc'")
            })?,
            filter_layout: filter_layout.unwrap_or("iohw").parse().map_err(|_| {
                PyValueError::new_err("Invalid filter_layout: expected 'iohw', 'hwoi', or 'ohwi'")
            })?,
            bias: bias.map(MLOperand::rustnn_index),
            output_padding: output_padding.unwrap_or_else(|| vec![0, 0]),
            output_sizes,
        };
        self.call(|builder| builder.conv_transpose2d_with_options(input, filter, options))
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
    #[pyo3(signature = (input, window_dimensions=None, strides=None, dilations=None, pads=None, layout=None, output_shape_rounding=None, output_sizes=None, *, label=None))]
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
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let mut options = make_pool2d_options(
            window_dimensions,
            strides.unwrap_or_else(|| vec![1; 2]),
            dilations.unwrap_or_else(|| vec![1; 2]),
            pads.unwrap_or_else(|| vec![0; 4]),
            parse_pool_layout(layout)?,
            output_shape_rounding,
            output_sizes,
        );
        options.label = label.unwrap_or_default().to_string();
        self.call(|builder| builder.average_pool2d_with_options(input, options))
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
    #[pyo3(signature = (input, window_dimensions=None, strides=None, dilations=None, pads=None, layout=None, output_shape_rounding=None, output_sizes=None, *, label=None))]
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
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let mut options = make_pool2d_options(
            window_dimensions,
            strides.unwrap_or_else(|| vec![1; 2]),
            dilations.unwrap_or_else(|| vec![1; 2]),
            pads.unwrap_or_else(|| vec![0; 4]),
            parse_pool_layout(layout)?,
            output_shape_rounding,
            output_sizes,
        );
        options.label = label.unwrap_or_default().to_string();
        self.call(|builder| builder.max_pool2d_with_options(input, options))
    }

    /// 2D L2 Pooling operation
    #[pyo3(signature = (input, window_dimensions=None, strides=None, dilations=None, pads=None, layout=None, output_shape_rounding=None, output_sizes=None, *, label=None))]
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
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let mut options = make_pool2d_options(
            window_dimensions,
            strides.unwrap_or_else(|| vec![1; 2]),
            dilations.unwrap_or_else(|| vec![1; 2]),
            pads.unwrap_or_else(|| vec![0; 4]),
            parse_pool_layout(layout)?,
            output_shape_rounding,
            output_sizes,
        );
        options.label = label.unwrap_or_default().to_string();
        self.call(|builder| builder.l2_pool2d_with_options(input, options))
    }

    /// Global Average Pooling operation
    ///
    /// Args:
    ///     input: Input operand (4D tensor)
    ///     layout: Input layout "nchw" or "nhwc" (default: "nchw")
    ///
    /// Returns:
    ///     MLOperand: The output operand with spatial dimensions reduced to 1x1
    #[pyo3(signature = (input, layout=None, *, label=None))]
    fn global_average_pool(
        &mut self,
        input: &PyMLOperand,
        layout: Option<&str>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let options = MLPool2dOptions {
            label: label.unwrap_or_default().to_string(),
            layout: parse_pool_layout(layout)?.to_string(),
            ..Default::default()
        };
        self.call(|builder| builder.global_average_pool_with_options(input, options))
    }

    /// Global Max Pooling operation
    ///
    /// Args:
    ///     input: Input operand (4D tensor)
    ///     layout: Input layout "nchw" or "nhwc" (default: "nchw")
    ///
    /// Returns:
    ///     MLOperand: The output operand with spatial dimensions reduced to 1x1
    #[pyo3(signature = (input, layout=None, *, label=None))]
    fn global_max_pool(
        &mut self,
        input: &PyMLOperand,
        layout: Option<&str>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let options = MLPool2dOptions {
            label: label.unwrap_or_default().to_string(),
            layout: parse_pool_layout(layout)?.to_string(),
            ..Default::default()
        };
        self.call(|builder| builder.global_max_pool_with_options(input, options))
    }

    // Normalization operations

    /// Batch Normalization operation
    #[pyo3(signature = (input, mean, variance, scale=None, bias=None, epsilon=1e-5, axis=1, *, label=None))]
    fn batch_normalization(
        &mut self,
        input: &PyMLOperand,
        mean: &PyMLOperand,
        variance: &PyMLOperand,
        scale: Option<&PyMLOperand>,
        bias: Option<&PyMLOperand>,
        epsilon: f64,
        axis: u32,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let mean = self.handle(mean)?;
        let variance = self.handle(variance)?;
        let scale = scale.map(|operand| self.handle(operand)).transpose()?;
        let bias = bias.map(|operand| self.handle(operand)).transpose()?;
        let options = MLBatchNormalizationOptions {
            label: label.unwrap_or_default().to_string(),
            scale: scale.map(MLOperand::rustnn_index),
            bias: bias.map(MLOperand::rustnn_index),
            epsilon,
            axis,
        };
        self.call(|builder| {
            builder.batch_normalization_with_options(input, mean, variance, options)
        })
    }

    /// Instance Normalization operation
    #[pyo3(signature = (input, scale=None, bias=None, epsilon=1e-5, layout=None, *, label=None))]
    fn instance_normalization(
        &mut self,
        input: &PyMLOperand,
        scale: Option<&PyMLOperand>,
        bias: Option<&PyMLOperand>,
        epsilon: f64,
        layout: Option<&str>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let scale = scale.map(|operand| self.handle(operand)).transpose()?;
        let bias = bias.map(|operand| self.handle(operand)).transpose()?;
        let options = MLInstanceNormalizationOptions {
            label: label.unwrap_or_default().to_string(),
            scale: scale.map(MLOperand::rustnn_index),
            bias: bias.map(MLOperand::rustnn_index),
            epsilon,
            layout: parse_pool_layout(layout)?.to_string(),
        };
        self.call(|builder| builder.instance_normalization_with_options(input, options))
    }

    /// Layer Normalization operation
    #[pyo3(signature = (input, scale=None, bias=None, epsilon=1e-5, axes=None, *, label=None))]
    fn layer_normalization(
        &mut self,
        input: &PyMLOperand,
        scale: Option<&PyMLOperand>,
        bias: Option<&PyMLOperand>,
        epsilon: f64,
        axes: Option<Vec<u32>>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let scale = scale.map(|operand| self.handle(operand)).transpose()?;
        let bias = bias.map(|operand| self.handle(operand)).transpose()?;
        let options = MLLayerNormalizationOptions {
            label: label.unwrap_or_default().to_string(),
            scale: scale.map(MLOperand::rustnn_index),
            bias: bias.map(MLOperand::rustnn_index),
            epsilon,
            axes,
        };
        self.call(|builder| builder.layer_normalization_with_options(input, options))
    }

    // Unary operations

    /// ReLU activation
    #[pyo3(signature = (x, *, label=None))]
    fn relu(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.relu_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Sigmoid activation
    #[pyo3(signature = (x, *, label=None))]
    fn sigmoid(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.sigmoid_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Tanh activation
    #[pyo3(signature = (x, *, label=None))]
    fn tanh(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.tanh_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Softmax activation
    ///
    /// Args:
    ///     x: Input operand
    ///     axis: Axis along which to normalize
    ///
    /// Returns:
    ///     MLOperand: Output operand with the same shape as input
    #[pyo3(signature = (x, axis, *, label=None))]
    fn softmax(
        &mut self,
        x: &PyMLOperand,
        axis: u32,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.softmax_with_options(
                x,
                axis,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    // Element-wise operations - Basic math

    /// Element-wise absolute value
    #[pyo3(signature = (x, *, label=None))]
    fn abs(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.abs_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise ceiling (round up)
    #[pyo3(signature = (x, *, label=None))]
    fn ceil(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.ceil_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise floor (round down)
    #[pyo3(signature = (x, *, label=None))]
    fn floor(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.floor_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise round-to-nearest-even (banker's rounding)
    #[pyo3(signature = (x, *, label=None))]
    fn round_even(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.round_even_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise negation
    #[pyo3(signature = (x, *, label=None))]
    fn neg(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.neg_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise sign (-1, 0, 1)
    #[pyo3(signature = (x, *, label=None))]
    fn sign(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.sign_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    // Element-wise operations - Exponential and logarithmic

    /// Element-wise natural exponential (e^x)
    #[pyo3(signature = (x, *, label=None))]
    fn exp(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.exp_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise natural logarithm
    #[pyo3(signature = (x, *, label=None))]
    fn log(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.log_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise square root
    #[pyo3(signature = (x, *, label=None))]
    fn sqrt(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.sqrt_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise reciprocal (1/x)
    #[pyo3(signature = (x, *, label=None))]
    fn reciprocal(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.reciprocal_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    // Element-wise operations - Trigonometric

    /// Element-wise sine
    #[pyo3(signature = (x, *, label=None))]
    fn sin(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.sin_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise cosine
    #[pyo3(signature = (x, *, label=None))]
    fn cos(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.cos_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise tangent
    #[pyo3(signature = (x, *, label=None))]
    fn tan(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.tan_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    // Element-wise operations - Special functions

    /// Element-wise error function
    #[pyo3(signature = (x, *, label=None))]
    fn erf(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.erf_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Identity operation (returns input unchanged)
    #[pyo3(signature = (x, *, label=None))]
    fn identity(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.identity_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    // Logic operations

    /// Element-wise equality comparison
    #[pyo3(signature = (a, b, *, label=None))]
    fn equal(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let a = self.handle(a)?;
        let b = self.handle(b)?;

        self.call(|builder| {
            builder.equal_with_options(
                a,
                b,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise greater than comparison
    #[pyo3(signature = (a, b, *, label=None))]
    fn greater(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let a = self.handle(a)?;
        let b = self.handle(b)?;

        self.call(|builder| {
            builder.greater_with_options(
                a,
                b,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise greater than or equal comparison
    #[pyo3(signature = (a, b, *, label=None))]
    fn greater_or_equal(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let a = self.handle(a)?;
        let b = self.handle(b)?;

        self.call(|builder| {
            builder.greater_or_equal_with_options(
                a,
                b,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise less than comparison
    #[pyo3(signature = (a, b, *, label=None))]
    fn lesser(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let a = self.handle(a)?;
        let b = self.handle(b)?;

        self.call(|builder| {
            builder.lesser_with_options(
                a,
                b,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise less than or equal comparison
    #[pyo3(signature = (a, b, *, label=None))]
    fn lesser_or_equal(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let a = self.handle(a)?;
        let b = self.handle(b)?;

        self.call(|builder| {
            builder.lesser_or_equal_with_options(
                a,
                b,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise not-equal comparison
    #[pyo3(signature = (a, b, *, label=None))]
    fn not_equal(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let a = self.handle(a)?;
        let b = self.handle(b)?;

        self.call(|builder| {
            builder.not_equal_with_options(
                a,
                b,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise logical NOT
    #[pyo3(signature = (x, *, label=None))]
    fn logical_not(&mut self, x: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.logical_not_with_options(
                x,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise logical AND
    #[pyo3(signature = (a, b, *, label=None))]
    fn logical_and(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let a = self.handle(a)?;
        let b = self.handle(b)?;

        self.call(|builder| {
            builder.logical_and_with_options(
                a,
                b,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise logical OR
    #[pyo3(signature = (a, b, *, label=None))]
    fn logical_or(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let a = self.handle(a)?;
        let b = self.handle(b)?;

        self.call(|builder| {
            builder.logical_or_with_options(
                a,
                b,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise logical XOR
    #[pyo3(signature = (a, b, *, label=None))]
    fn logical_xor(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let a = self.handle(a)?;
        let b = self.handle(b)?;

        self.call(|builder| {
            builder.logical_xor_with_options(
                a,
                b,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// DequantizeLinear operation
    /// Converts quantized integer values to floating-point representation
    /// Formula: output = (input - zeroPoint) * scale
    #[pyo3(signature = (input, scale, zero_point=None, *, label=None))]
    fn dequantize_linear(
        &mut self,
        input: &PyMLOperand,
        scale: &PyMLOperand,
        zero_point: Option<&PyMLOperand>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let scale = self.handle(scale)?;
        let zero_point = zero_point.map(|operand| self.handle(operand)).transpose()?;

        self.call(|builder| {
            builder.dequantize_linear_with_options(
                input,
                scale,
                zero_point,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// QuantizeLinear operation
    /// Converts floating-point values to quantized integer representation
    /// Formula: output = input / scale + zeroPoint
    #[pyo3(signature = (input, scale, zero_point=None, *, label=None))]
    fn quantize_linear(
        &mut self,
        input: &PyMLOperand,
        scale: &PyMLOperand,
        zero_point: Option<&PyMLOperand>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let scale = self.handle(scale)?;
        let zero_point = zero_point.map(|operand| self.handle(operand)).transpose()?;

        self.call(|builder| {
            builder.quantize_linear_with_options(
                input,
                scale,
                zero_point,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Reshape operation
    #[pyo3(signature = (x, new_shape, *, label=None))]
    fn reshape(
        &mut self,
        x: &PyMLOperand,
        new_shape: &Bound<'_, PyAny>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let new_shape = parse_dimensions(new_shape)?;
        let x = self.handle(x)?;

        self.call(|builder| {
            builder.reshape_with_options(
                x,
                new_shape,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
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
    #[pyo3(signature = (input, axes=None, keep_dimensions=false, *, label=None))]
    fn reduce_sum(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.reduce_sum_with_options(
                input,
                MLReduceOptions {
                    label: label.unwrap_or_default().to_string(),
                    axes,
                    keep_dimensions,
                },
            )
        })
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
    #[pyo3(signature = (input, axes=None, keep_dimensions=false, *, label=None))]
    fn reduce_mean(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.reduce_mean_with_options(
                input,
                MLReduceOptions {
                    label: label.unwrap_or_default().to_string(),
                    axes,
                    keep_dimensions,
                },
            )
        })
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
    #[pyo3(signature = (input, axes=None, keep_dimensions=false, *, label=None))]
    fn reduce_max(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.reduce_max_with_options(
                input,
                MLReduceOptions {
                    label: label.unwrap_or_default().to_string(),
                    axes,
                    keep_dimensions,
                },
            )
        })
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
    #[pyo3(signature = (input, axes=None, keep_dimensions=false, *, label=None))]
    fn reduce_min(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.reduce_min_with_options(
                input,
                MLReduceOptions {
                    label: label.unwrap_or_default().to_string(),
                    axes,
                    keep_dimensions,
                },
            )
        })
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
    #[pyo3(signature = (input, axes=None, keep_dimensions=false, *, label=None))]
    fn reduce_product(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.reduce_product_with_options(
                input,
                MLReduceOptions {
                    label: label.unwrap_or_default().to_string(),
                    axes,
                    keep_dimensions,
                },
            )
        })
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
    #[pyo3(signature = (input, axes=None, keep_dimensions=false, *, label=None))]
    fn reduce_l1(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.reduce_l1_with_options(
                input,
                MLReduceOptions {
                    label: label.unwrap_or_default().to_string(),
                    axes,
                    keep_dimensions,
                },
            )
        })
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
    #[pyo3(signature = (input, axes=None, keep_dimensions=false, *, label=None))]
    fn reduce_l2(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.reduce_l2_with_options(
                input,
                MLReduceOptions {
                    label: label.unwrap_or_default().to_string(),
                    axes,
                    keep_dimensions,
                },
            )
        })
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
    #[pyo3(signature = (input, axes=None, keep_dimensions=false, *, label=None))]
    fn reduce_log_sum(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.reduce_log_sum_with_options(
                input,
                MLReduceOptions {
                    label: label.unwrap_or_default().to_string(),
                    axes,
                    keep_dimensions,
                },
            )
        })
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
    #[pyo3(signature = (input, axes=None, keep_dimensions=false, *, label=None))]
    fn reduce_log_sum_exp(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.reduce_log_sum_exp_with_options(
                input,
                MLReduceOptions {
                    label: label.unwrap_or_default().to_string(),
                    axes,
                    keep_dimensions,
                },
            )
        })
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
    #[pyo3(signature = (input, axes=None, keep_dimensions=false, *, label=None))]
    fn reduce_sum_square(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.reduce_sum_square_with_options(
                input,
                MLReduceOptions {
                    label: label.unwrap_or_default().to_string(),
                    axes,
                    keep_dimensions,
                },
            )
        })
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
    #[pyo3(signature = (input, permutation=None, *, label=None))]
    fn transpose(
        &mut self,
        input: &PyMLOperand,
        permutation: Option<Vec<u32>>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.transpose_with_options(
                input,
                MLTransposeOptions {
                    label: label.unwrap_or_default().to_string(),
                    permutation: permutation.unwrap_or_default(),
                },
            )
        })
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
    #[pyo3(signature = (inputs, axis, *, label=None))]
    fn concat(
        &mut self,
        inputs: Vec<PyMLOperand>,
        axis: u32,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let inputs: Vec<_> = inputs
            .iter()
            .map(|operand| self.handle(operand))
            .collect::<PyResult<_>>()?;

        self.call(|builder| {
            builder.concat_with_options(
                &inputs,
                axis,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
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
    #[pyo3(signature = (input, starts, sizes, strides=None, *, label=None))]
    fn slice(
        &mut self,
        input: &PyMLOperand,
        starts: Vec<u32>,
        sizes: &Bound<'_, PyAny>,
        strides: Option<Vec<i32>>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let sizes = parse_dimensions(sizes)?;
        let input = self.handle(input)?;
        let options = MLSliceOptions {
            label: label.unwrap_or_default().to_string(),
            strides: strides
                .unwrap_or_else(|| vec![1; starts.len()])
                .into_iter()
                .map(|stride| u32::try_from(stride).map_err(builder_error))
                .collect::<PyResult<_>>()?,
        };
        self.call(|builder| builder.slice_with_options(input, &starts, &sizes, options))
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
    #[pyo3(signature = (input, new_shape, *, label=None))]
    fn expand(
        &mut self,
        input: &PyMLOperand,
        new_shape: &Bound<'_, PyAny>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let new_shape = parse_dimensions(new_shape)?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.expand_with_options(
                input,
                new_shape,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
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
    #[pyo3(signature = (input, indices, axis=0, *, label=None))]
    fn gather(
        &mut self,
        input: &PyMLOperand,
        indices: &PyMLOperand,
        axis: u32,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let indices = self.handle(indices)?;

        self.call(|builder| {
            builder.gather_with_options(
                input,
                indices,
                MLGatherOptions {
                    label: label.unwrap_or_default().to_string(),
                    axis,
                },
            )
        })
    }

    /// Gather elements operation (output shape equals indices shape)
    #[pyo3(signature = (input, indices, axis=0, *, label=None))]
    fn gather_elements(
        &mut self,
        input: &PyMLOperand,
        indices: &PyMLOperand,
        axis: u32,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let indices = self.handle(indices)?;

        self.call(|builder| {
            builder.gather_elements_with_options(
                input,
                indices,
                MLGatherOptions {
                    label: label.unwrap_or_default().to_string(),
                    axis,
                },
            )
        })
    }

    /// Gather ND operation
    #[pyo3(signature = (input, indices, *, label=None))]
    fn gather_nd(
        &mut self,
        input: &PyMLOperand,
        indices: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let indices = self.handle(indices)?;

        self.call(|builder| {
            builder.gather_nd_with_options(
                input,
                indices,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
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
    #[pyo3(signature = (input, splits, axis=0, *, label=None))]
    fn split(
        &mut self,
        _py: Python,
        input: &PyMLOperand,
        splits: &Bound<'_, PyAny>,
        axis: u32,
        label: Option<&str>,
    ) -> PyResult<Vec<PyMLOperand>> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let handles = if let Ok(count) = splits.extract::<u32>() {
            self.inner_mut()?.split_equal_with_options(
                input,
                count,
                MLSplitOptions {
                    label: label.unwrap_or_default().to_string(),
                    axis,
                },
            )
        } else if let Ok(sizes) = splits.extract::<Vec<u32>>() {
            self.inner_mut()?.split_with_options(
                input,
                &sizes,
                MLSplitOptions {
                    label: label.unwrap_or_default().to_string(),
                    axis,
                },
            )
        } else {
            return Err(PyValueError::new_err(
                "splits must be either an integer or a list of integers",
            ));
        }
        .map_err(builder_error)?;
        self.wrap_many(handles)
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
    #[pyo3(signature = (condition, true_value, false_value, *, label=None))]
    fn where_(
        &mut self,
        condition: &PyMLOperand,
        true_value: &PyMLOperand,
        false_value: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let condition = self.handle(condition)?;
        let true_value = self.handle(true_value)?;
        let false_value = self.handle(false_value)?;

        self.call(|builder| {
            builder.where_with_options(
                condition,
                true_value,
                false_value,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
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
    #[pyo3(signature = (input, padding, mode=None, value=None, *, label=None))]
    fn pad(
        &mut self,
        py: Python<'_>,
        input: &PyMLOperand,
        padding: Vec<u32>,
        mode: Option<&str>,
        value: Option<Py<PyAny>>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let half = padding.len() / 2;
        let beginning = padding[..half].to_vec();
        let ending = padding[half..].to_vec();
        let options = MLPadOptions {
            label: label.unwrap_or_default().to_string(),
            mode: mode.unwrap_or("constant").parse().map_err(|_| {
                PyValueError::new_err("Invalid mode: expected 'constant', 'edge', or 'reflection'")
            })?,
            value: pad_options_value(py, value)?,
        };
        self.call(|builder| builder.pad_with_options(input, beginning, ending, options))
    }

    /// GELU activation operation
    ///
    /// Args:
    ///     input: The input tensor
    ///
    /// Returns:
    ///     MLOperand: Output operand with GELU activation applied
    #[pyo3(signature = (input, *, label=None))]
    fn gelu(&mut self, input: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.gelu_with_options(
                input,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Squeeze operation (remove dimensions of size 1)
    ///
    /// Args:
    ///     input: The input tensor
    ///     axes: Optional sequence of axes to squeeze. If None, all dimensions of size 1 are removed
    ///
    /// Returns:
    ///     MLOperand: Output operand with dimensions squeezed
    #[pyo3(signature = (input, axes=None, *, label=None))]
    fn squeeze(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.squeeze_with_options(
                input,
                MLSqueezeOptions {
                    label: label.unwrap_or_default().to_string(),
                    axes: axes.unwrap_or_default(),
                },
            )
        })
    }

    /// Unsqueeze operation (add dimensions of size 1)
    ///
    /// Args:
    ///     input: The input tensor
    ///     axes: Sequence of axes where dimensions of size 1 should be inserted
    ///
    /// Returns:
    ///     MLOperand: Output operand with dimensions unsqueezed
    #[pyo3(signature = (input, axes=None, *, label=None))]
    fn unsqueeze(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.unsqueeze_with_options(
                input,
                MLUnsqueezeOptions {
                    label: label.unwrap_or_default().to_string(),
                    axes: axes.unwrap_or_default(),
                },
            )
        })
    }

    /// ArgMax operation (find indices of maximum values)
    ///
    /// Args:
    ///     input: The input tensor
    ///     axis: The axis to reduce along
    ///     keep_dimensions: If True, keep the reduced axis with size 1. Default is False
    ///     output_data_type: Output data type for indices ("int32" or "int64"). Default is "int32"
    ///
    /// Returns:
    ///     MLOperand: Output operand containing indices of maximum values
    #[pyo3(signature = (input, axis, keep_dimensions=false, output_data_type=None, *, label=None))]
    fn arg_max(
        &mut self,
        input: &PyMLOperand,
        axis: u32,
        keep_dimensions: bool,
        output_data_type: Option<&str>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let output_data_type = match output_data_type {
            Some("int32") | None => MLOperandDataType::Int32,
            Some("int64") => MLOperandDataType::Int64,
            _ => {
                return Err(PyValueError::new_err(
                    "output_data_type must be 'int32' or 'int64'",
                ))
            }
        };
        let options = MLArgMinMaxOptions {
            label: label.unwrap_or_default().to_string(),
            keep_dimensions,
            output_data_type,
        };
        self.call(|builder| builder.arg_max_with_options(input, axis, options))
    }

    /// ArgMin operation (find indices of minimum values)
    ///
    /// Args:
    ///     input: The input tensor
    ///     axis: The axis to reduce along
    ///     keep_dimensions: If True, keep the reduced axis with size 1. Default is False
    ///     output_data_type: Output data type for indices ("int32" or "int64"). Default is "int32"
    ///
    /// Returns:
    ///     MLOperand: Output operand containing indices of minimum values
    #[pyo3(signature = (input, axis, keep_dimensions=false, output_data_type=None, *, label=None))]
    fn arg_min(
        &mut self,
        input: &PyMLOperand,
        axis: u32,
        keep_dimensions: bool,
        output_data_type: Option<&str>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let output_data_type = match output_data_type {
            Some("int32") | None => MLOperandDataType::Int32,
            Some("int64") => MLOperandDataType::Int64,
            _ => {
                return Err(PyValueError::new_err(
                    "output_data_type must be 'int32' or 'int64'",
                ))
            }
        };
        let options = MLArgMinMaxOptions {
            label: label.unwrap_or_default().to_string(),
            keep_dimensions,
            output_data_type,
        };
        self.call(|builder| builder.arg_min_with_options(input, axis, options))
    }

    /// Cast operation (type conversion)
    ///
    /// Args:
    ///     input: The input tensor
    ///     data_type: Target data type ("float32", "float16", "int32", "uint32", "int8", "uint8", "int64")
    ///
    /// Returns:
    ///     MLOperand: Output operand with converted type
    #[pyo3(signature = (input, data_type, *, label=None))]
    fn cast(
        &mut self,
        input: &PyMLOperand,
        data_type: &str,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let data_type = data_type_to_ml_operand(parse_data_type(data_type)?)?;
        self.call(|builder| {
            builder.cast_with_options(
                input,
                data_type,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Save through RustNN as .webnn text plus sibling .safetensors weights.
    /// Saving borrows the native builder; call before build().
    fn save(&mut self, outputs: &Bound<'_, PyDict>, path: &str) -> PyResult<()> {
        let named = self.named_outputs(outputs)?;
        let outputs = named
            .iter()
            .map(|(name, operand)| (name.as_str(), *operand))
            .collect();
        self.inner_mut()?
            .rustnn_save_webnn(&outputs, path)
            .map_err(map_rustnn_error)
    }

    /// Build the computational graph
    ///
    /// Args:
    ///     outputs: Dictionary mapping output names to MLOperand objects
    ///
    /// Returns:
    ///     MLGraph: The compiled graph
    fn build(&mut self, py: Python<'_>, outputs: &Bound<'_, PyDict>) -> PyResult<PyMLGraph> {
        self.inner_mut()?;
        let named = self.named_outputs(outputs)?;
        let outputs = named
            .iter()
            .map(|(name, operand)| (name.as_str(), *operand))
            .collect();
        // Compilation consumes this Python builder even when finalization/backend compilation fails.
        let mut inner = self.inner.take().expect("checked above");
        let compiled = inner.build(&outputs);
        drop(inner);
        Ok(PyMLGraph::new(
            compiled.map_err(map_rustnn_error)?,
            self.context.clone_ref(py),
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
    #[pyo3(signature = (input, indices, updates, axis=0, *, label=None))]
    fn scatter_elements(
        &mut self,
        input: &PyMLOperand,
        indices: &PyMLOperand,
        updates: &PyMLOperand,
        axis: u32,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let indices = self.handle(indices)?;
        let updates = self.handle(updates)?;

        self.call(|builder| {
            builder.scatter_elements_with_options(
                input,
                indices,
                updates,
                MLScatterOptions {
                    label: label.unwrap_or_default().to_string(),
                    axis,
                },
            )
        })
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
    #[pyo3(signature = (input, indices, updates, *, label=None))]
    fn scatter_nd(
        &mut self,
        input: &PyMLOperand,
        indices: &PyMLOperand,
        updates: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let indices = self.handle(indices)?;
        let updates = self.handle(updates)?;

        self.call(|builder| {
            builder.scatter_nd_with_options(
                input,
                indices,
                updates,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
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
    #[pyo3(signature = (input, repetitions, *, label=None))]
    fn tile(
        &mut self,
        input: &PyMLOperand,
        repetitions: Vec<u32>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.tile_with_options(
                input,
                repetitions,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
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
    #[pyo3(signature = (input, upper=true, diagonal=0, *, label=None))]
    fn triangular(
        &mut self,
        input: &PyMLOperand,
        upper: bool,
        diagonal: i32,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.triangular_with_options(
                input,
                MLTriangularOptions {
                    label: label.unwrap_or_default().to_string(),
                    upper: Some(upper),
                    diagonal,
                },
            )
        })
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
    #[pyo3(signature = (input, alpha=0.2, beta=0.5, *, label=None))]
    fn hard_sigmoid(
        &mut self,
        input: &PyMLOperand,
        alpha: f64,
        beta: f64,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.hard_sigmoid_with_options(
                input,
                MLHardSigmoidOptions {
                    label: label.unwrap_or_default().to_string(),
                    alpha,
                    beta,
                },
            )
        })
    }

    /// hardSwish activation operation
    ///
    /// Computes element-wise: y = x * clamp(x + 3, 0, 6) / 6
    ///
    /// Args:
    ///     input: Input tensor
    ///
    /// Returns:
    ///     MLOperand: Output operand
    #[pyo3(signature = (input, *, label=None))]
    fn hard_swish(&mut self, input: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        self.call(|builder| {
            builder.hard_swish_with_options(
                input,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
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
    #[pyo3(signature = (input, *, label=None))]
    fn softplus(&mut self, input: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.softplus_with_options(
                input,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
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
    #[pyo3(signature = (input, *, label=None))]
    fn softsign(&mut self, input: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.softsign_with_options(
                input,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
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
    #[pyo3(signature = (input, min_value=None, max_value=None, *, label=None))]
    fn clamp(
        &mut self,
        py: Python<'_>,
        input: &PyMLOperand,
        min_value: Option<Py<PyAny>>,
        max_value: Option<Py<PyAny>>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let min_value = clamp_limit_to_json(py, min_value)?;
        let max_value = clamp_limit_to_json(py, max_value)?;
        let options = MLClampOptions {
            label: label.unwrap_or_default().to_string(),
            min_value,
            max_value,
        };
        self.call(|builder| builder.clamp_with_options(input, options))
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
    #[pyo3(signature = (input, alpha=1.0, *, label=None))]
    fn elu(
        &mut self,
        input: &PyMLOperand,
        alpha: f64,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.elu_with_options(
                input,
                MLEluOptions {
                    label: label.unwrap_or_default().to_string(),
                    alpha,
                },
            )
        })
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
    #[pyo3(signature = (input, alpha=0.01, *, label=None))]
    fn leaky_relu(
        &mut self,
        input: &PyMLOperand,
        alpha: f64,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.leaky_relu_with_options(
                input,
                MLLeakyReluOptions {
                    label: label.unwrap_or_default().to_string(),
                    alpha,
                },
            )
        })
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
    #[pyo3(signature = (input, slope, *, label=None))]
    fn prelu(
        &mut self,
        input: &PyMLOperand,
        slope: &PyMLOperand,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let slope = self.handle(slope)?;

        self.call(|builder| {
            builder.prelu_with_options(
                input,
                slope,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Linear activation: output = alpha * input + beta
    #[pyo3(signature = (input, alpha=1.0, beta=0.0, *, label=None))]
    fn linear(
        &mut self,
        input: &PyMLOperand,
        alpha: f64,
        beta: f64,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.linear_with_options(
                input,
                MLLinearOptions {
                    label: label.unwrap_or_default().to_string(),
                    alpha,
                    beta,
                },
            )
        })
    }

    /// Element-wise is-NaN test (uint8 output)
    #[pyo3(signature = (input, *, label=None))]
    fn is_nan(&mut self, input: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.is_nan_with_options(
                input,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Element-wise is-infinite test (uint8 output)
    #[pyo3(signature = (input, *, label=None))]
    fn is_infinite(&mut self, input: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.is_infinite_with_options(
                input,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Returns the rank of the input as a 1D int64 tensor
    #[pyo3(signature = (input, *, label=None))]
    fn shape(&mut self, input: &PyMLOperand, label: Option<&str>) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.shape_with_options(
                input,
                MLOperatorOptions {
                    label: label.unwrap_or_default().to_string(),
                },
            )
        })
    }

    /// Cumulative sum along an axis
    #[pyo3(signature = (input, axis, exclusive=false, reversed=false, *, label=None))]
    fn cumulative_sum(
        &mut self,
        input: &PyMLOperand,
        axis: u32,
        exclusive: bool,
        reversed: bool,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.cumulative_sum_with_options(
                input,
                axis,
                MLCumulativeSumOptions {
                    label: label.unwrap_or_default().to_string(),
                    exclusive,
                    reversed,
                },
            )
        })
    }

    /// Reverse tensor along axes (default: all axes)
    #[pyo3(signature = (input, axes=None, *, label=None))]
    fn reverse(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;

        self.call(|builder| {
            builder.reverse_with_options(
                input,
                MLReverseOptions {
                    label: label.unwrap_or_default().to_string(),
                    axes,
                },
            )
        })
    }

    /// 2D resampling (resize spatial axes)
    #[pyo3(signature = (input, sizes=None, scales=None, mode="nearest-neighbor", axes=None, *, label=None))]
    fn resample2d(
        &mut self,
        input: &PyMLOperand,
        sizes: Option<Vec<u32>>,
        scales: Option<Vec<f32>>,
        mode: &str,
        axes: Option<Vec<u32>>,
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let options = MLResample2dOptions {
            label: label.unwrap_or_default().to_string(),
            sizes,
            scales: scales.unwrap_or_default(),
            mode: mode.to_string(),
            axes: axes.unwrap_or_default(),
        };
        self.call(|builder| builder.resample2d_with_options(input, options))
    }

    /// GRU recurrent network
    #[pyo3(signature = (input, weight, recurrent_weight, steps, hidden_size, bias=None, recurrent_bias=None, initial_hidden_state=None, reset_after=false, return_sequence=false, direction="forward", layout="zrn", activations=None, *, label=None))]
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
        label: Option<&str>,
    ) -> PyResult<Vec<PyMLOperand>> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let weight = self.handle(weight)?;
        let recurrent_weight = self.handle(recurrent_weight)?;
        let bias = bias.map(|operand| self.handle(operand)).transpose()?;
        let recurrent_bias = recurrent_bias
            .map(|operand| self.handle(operand))
            .transpose()?;
        let initial_hidden_state = initial_hidden_state
            .map(|operand| self.handle(operand))
            .transpose()?;
        let options = MLGruOptions {
            label: label.unwrap_or_default().to_string(),
            bias: bias.map(MLOperand::rustnn_index),
            recurrent_bias: recurrent_bias.map(MLOperand::rustnn_index),
            initial_hidden_state: initial_hidden_state.map(MLOperand::rustnn_index),
            reset_after,
            return_sequence,
            direction: direction.to_string(),
            layout: layout.to_string(),
            activations,
        };
        self.call_many(|builder| {
            builder.gru_with_options(input, weight, recurrent_weight, steps, hidden_size, options)
        })
    }

    /// GRU cell (single step)
    #[pyo3(signature = (input, weight, recurrent_weight, hidden_state, hidden_size, bias=None, recurrent_bias=None, reset_after=false, layout="zrn", activations=None, *, label=None))]
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
        label: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let weight = self.handle(weight)?;
        let recurrent_weight = self.handle(recurrent_weight)?;
        let hidden_state = self.handle(hidden_state)?;
        let bias = bias.map(|operand| self.handle(operand)).transpose()?;
        let recurrent_bias = recurrent_bias
            .map(|operand| self.handle(operand))
            .transpose()?;
        let options = MLGruCellOptions {
            label: label.unwrap_or_default().to_string(),
            bias: bias.map(MLOperand::rustnn_index),
            recurrent_bias: recurrent_bias.map(MLOperand::rustnn_index),
            reset_after,
            layout: layout.to_string(),
            activations,
        };
        self.call(|builder| {
            builder.gru_cell_with_options(
                input,
                weight,
                recurrent_weight,
                hidden_state,
                hidden_size,
                options,
            )
        })
    }

    /// LSTM recurrent network
    #[pyo3(signature = (input, weight, recurrent_weight, steps, hidden_size, bias=None, recurrent_bias=None, peephole_weight=None, initial_hidden_state=None, initial_cell_state=None, return_sequence=false, direction="forward", layout="iofg", activations=None, *, label=None))]
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
        label: Option<&str>,
    ) -> PyResult<Vec<PyMLOperand>> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let weight = self.handle(weight)?;
        let recurrent_weight = self.handle(recurrent_weight)?;
        let bias = bias.map(|operand| self.handle(operand)).transpose()?;
        let recurrent_bias = recurrent_bias
            .map(|operand| self.handle(operand))
            .transpose()?;
        let peephole_weight = peephole_weight
            .map(|operand| self.handle(operand))
            .transpose()?;
        let initial_hidden_state = initial_hidden_state
            .map(|operand| self.handle(operand))
            .transpose()?;
        let initial_cell_state = initial_cell_state
            .map(|operand| self.handle(operand))
            .transpose()?;
        let options = MLLstmOptions {
            label: label.unwrap_or_default().to_string(),
            bias: bias.map(MLOperand::rustnn_index),
            recurrent_bias: recurrent_bias.map(MLOperand::rustnn_index),
            peephole_weight: peephole_weight.map(MLOperand::rustnn_index),
            initial_hidden_state: initial_hidden_state.map(MLOperand::rustnn_index),
            initial_cell_state: initial_cell_state.map(MLOperand::rustnn_index),
            return_sequence,
            direction: direction.to_string(),
            layout: layout.to_string(),
            activations,
        };
        self.call_many(|builder| {
            builder.lstm_with_options(input, weight, recurrent_weight, steps, hidden_size, options)
        })
    }

    /// LSTM cell (single step)
    #[pyo3(signature = (input, weight, recurrent_weight, hidden_state, cell_state, hidden_size, bias=None, recurrent_bias=None, peephole_weight=None, layout="iofg", activations=None, *, label=None))]
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
        label: Option<&str>,
    ) -> PyResult<Vec<PyMLOperand>> {
        self.inner_mut()?;
        let input = self.handle(input)?;
        let weight = self.handle(weight)?;
        let recurrent_weight = self.handle(recurrent_weight)?;
        let hidden_state = self.handle(hidden_state)?;
        let cell_state = self.handle(cell_state)?;
        let bias = bias.map(|operand| self.handle(operand)).transpose()?;
        let recurrent_bias = recurrent_bias
            .map(|operand| self.handle(operand))
            .transpose()?;
        let peephole_weight = peephole_weight
            .map(|operand| self.handle(operand))
            .transpose()?;
        let options = MLLstmCellOptions {
            label: label.unwrap_or_default().to_string(),
            bias: bias.map(MLOperand::rustnn_index),
            recurrent_bias: recurrent_bias.map(MLOperand::rustnn_index),
            peephole_weight: peephole_weight.map(MLOperand::rustnn_index),
            layout: layout.to_string(),
            activations,
        };
        self.call_many(|builder| {
            builder.lstm_cell_with_options(
                input,
                weight,
                recurrent_weight,
                hidden_state,
                cell_state,
                hidden_size,
                options,
            )
        })
    }

    /// Query the static/max shape of an operand without building the graph
    fn operand_shape(&mut self, operand: &PyMLOperand) -> PyResult<Vec<u64>> {
        let handle = self.handle(operand)?;
        self.inner_mut()?
            .rustnn_operand_shape(handle)
            .map_err(map_rustnn_error)
    }

    /// Query the data type of an operand
    fn operand_data_type(&mut self, operand: &PyMLOperand) -> PyResult<String> {
        let handle = self.handle(operand)?;
        Ok(self
            .inner_mut()?
            .rustnn_operand_data_type(handle)
            .map_err(map_rustnn_error)?
            .as_str()
            .to_string())
    }
}

impl PyMLGraphBuilder {
    fn named_outputs(&self, outputs: &Bound<'_, PyDict>) -> PyResult<BTreeMap<String, MLOperand>> {
        outputs
            .iter()
            .map(|(name, value)| {
                let operand: PyMLOperand = value.extract()?;
                Ok((name.extract()?, self.handle(&operand)?))
            })
            .collect()
    }

    pub(crate) fn new_for_context(
        py: Python<'_>,
        context: Py<super::context::PyMLContext>,
    ) -> PyResult<Self> {
        static NEXT_BUILDER: AtomicU64 = AtomicU64::new(1);
        let inner = context.bind(py).borrow().create_runtime_builder()?;
        Ok(Self {
            inner: Some(inner),
            context,
            builder_id: NEXT_BUILDER.fetch_add(1, Ordering::Relaxed),
        })
    }

    fn inner_mut(&mut self) -> PyResult<&mut RuntimeBuilder> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("MLGraphBuilder.build() was already called"))
    }

    fn check_owner(&self, operand: &PyMLOperand) -> PyResult<()> {
        if operand.builder_id != self.builder_id {
            return Err(PyValueError::new_err(
                "Operand belongs to a different MLGraphBuilder",
            ));
        }
        Ok(())
    }

    fn handle(&self, operand: &PyMLOperand) -> PyResult<MLOperand> {
        self.check_owner(operand)?;
        Ok(operand.handle)
    }

    fn wrap(&self, handle: MLOperand) -> PyResult<PyMLOperand> {
        Ok(PyMLOperand {
            handle,
            builder_id: self.builder_id,
        })
    }

    fn wrap_many(&mut self, handles: Vec<MLOperand>) -> PyResult<Vec<PyMLOperand>> {
        handles
            .into_iter()
            .map(|handle| self.wrap(handle))
            .collect()
    }

    fn call<E: std::fmt::Display>(
        &mut self,
        op: impl FnOnce(&mut RuntimeBuilder) -> Result<MLOperand, E>,
    ) -> PyResult<PyMLOperand> {
        let handle = op(self.inner_mut()?).map_err(builder_error)?;
        self.wrap(handle)
    }

    fn call_many<E: std::fmt::Display>(
        &mut self,
        op: impl FnOnce(&mut RuntimeBuilder) -> Result<Vec<MLOperand>, E>,
    ) -> PyResult<Vec<PyMLOperand>> {
        let handles = op(self.inner_mut()?).map_err(builder_error)?;
        self.wrap_many(handles)
    }
}
