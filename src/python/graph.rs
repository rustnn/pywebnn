//! Compiled computational graph representation
//!
//! PyO3 macros generate unsafe code that triggers unsafe_op_in_unsafe_fn warnings.
//! This is expected behavior from the macro-generated code.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::useless_conversion)]

use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use rustnn::graph::GraphInfo;
use rustnn::webnn_json;
use std::fs;
use std::path::Path;

/// Represents a compiled computational graph
#[pyclass(name = "MLGraph")]
pub struct PyMLGraph {
    pub(crate) graph_info: GraphInfo,
}

#[pymethods]
impl PyMLGraph {
    fn __repr__(&self) -> String {
        format!(
            "MLGraph(operands={}, operations={})",
            self.graph_info.operands.len(),
            self.graph_info.operations.len()
        )
    }

    /// Get the number of operands in the graph
    #[getter]
    fn operand_count(&self) -> usize {
        self.graph_info.operands.len()
    }

    /// Get the number of operations in the graph
    #[getter]
    fn operation_count(&self) -> usize {
        self.graph_info.operations.len()
    }

    /// Get input names
    fn get_input_names(&self) -> Vec<String> {
        self.graph_info
            .operands
            .iter()
            .filter_map(|op| {
                if matches!(op.kind, rustnn::graph::OperandKind::Input) {
                    op.name.clone()
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get output names
    fn get_output_names(&self) -> Vec<String> {
        // Use output_operands list instead of filtering by kind
        self.graph_info
            .output_operands
            .iter()
            .filter_map(|&idx| {
                self.graph_info
                    .operands
                    .get(idx as usize)
                    .and_then(|op| op.name.clone())
            })
            .collect()
    }

    /// Debug method to inspect operand at index (for debugging)
    fn debug_operand(&self, idx: usize) -> String {
        if let Some(op) = self.graph_info.operands.get(idx) {
            format!(
                "Operand[{}]: name={:?}, kind={:?}, type={:?}, shape={:?}",
                idx, op.name, op.kind, op.descriptor.data_type, op.descriptor.shape
            )
        } else {
            format!("Operand[{}]: not found", idx)
        }
    }

    /// Count operands with empty shapes
    fn count_empty_shapes(&self) -> usize {
        self.graph_info
            .operands
            .iter()
            .filter(|op| op.descriptor.shape.is_empty())
            .count()
    }

    /// Count operands with empty shapes that are not constants (likely unknown shapes).
    fn count_unknown_shapes(&self) -> usize {
        self.graph_info
            .operands
            .iter()
            .filter(|op| {
                op.descriptor.shape.is_empty()
                    && !matches!(op.kind, rustnn::graph::OperandKind::Constant)
            })
            .count()
    }

    /// Count constant operands with empty shapes (scalar constants).
    fn count_scalar_constants(&self) -> usize {
        self.graph_info
            .operands
            .iter()
            .filter(|op| {
                op.descriptor.shape.is_empty()
                    && matches!(op.kind, rustnn::graph::OperandKind::Constant)
            })
            .count()
    }

    /// Count unknown shapes, excluding outputs that are known to be scalar from reduction ops.
    fn count_unknown_shapes_excluding_scalar_ops(&self) -> usize {
        use std::collections::HashSet;

        fn parse_i64_array(value: &serde_json::Value) -> Option<Vec<i64>> {
            let arr = value.as_array()?;
            let mut out = Vec::with_capacity(arr.len());
            for v in arr {
                if let Some(n) = v.as_i64() {
                    out.push(n);
                } else if let Some(n) = v.as_u64() {
                    out.push(n as i64);
                } else {
                    return None;
                }
            }
            Some(out)
        }

        let mut scalar_outputs = HashSet::new();
        for op in &self.graph_info.operations {
            let op_type = op.op_type().to_ascii_lowercase();
            if op_type == "constant" {
                for &output_id in op.output_operands_slice() {
                    scalar_outputs.insert(output_id);
                }
                continue;
            }
            if !matches!(
                op_type.as_str(),
                "reducemean"
                    | "reducesum"
                    | "reducemax"
                    | "reducemin"
                    | "reduceproduct"
                    | "reducel1"
                    | "reducel2"
                    | "reducelogsum"
                    | "reducelogsumexp"
                    | "reducesumsquare"
            ) {
                continue;
            }

            let attrs = op.attributes_json_value();
            let keep_dimensions = attrs
                .get("keepDimensions")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if keep_dimensions {
                continue;
            }

            let Some(output_id) = op.output_operand() else {
                continue;
            };
            let Some(input_id) = op.input_operands().first().copied() else {
                continue;
            };
            let input_shape = &self.graph_info.operands[input_id as usize].descriptor.shape;
            if input_shape.is_empty() {
                continue;
            }

            let axes = attrs.get("axes").and_then(parse_i64_array);
            let Some(axes) = axes else {
                continue;
            };
            let rank = input_shape.len() as i64;
            let mut normalized = HashSet::new();
            let mut valid = true;
            for axis in axes {
                let mut axis = axis;
                if axis < 0 {
                    axis += rank;
                }
                if axis < 0 || axis >= rank {
                    valid = false;
                    break;
                }
                normalized.insert(axis as usize);
            }
            if valid && normalized.len() == input_shape.len() {
                scalar_outputs.insert(output_id);
            }
        }

        self.graph_info
            .operands
            .iter()
            .enumerate()
            .filter(|(idx, op)| {
                op.descriptor.shape.is_empty()
                    && !matches!(op.kind, rustnn::graph::OperandKind::Constant)
                    && !scalar_outputs.contains(&(*idx as u32))
            })
            .count()
    }

    /// Debug unknown shapes with producer op and input shapes.
    fn debug_unknown_shapes(&self) -> Vec<String> {
        use std::collections::HashMap;

        let mut producer: HashMap<u32, (String, Vec<u32>)> = HashMap::new();
        for op in &self.graph_info.operations {
            let op_type = op.op_type().to_string();
            let input_ids = op.input_operands();
            for &output_id in op.output_operands_slice() {
                producer.insert(output_id, (op_type.clone(), input_ids.clone()));
            }
        }

        let mut out = Vec::new();
        for (idx, operand) in self.graph_info.operands.iter().enumerate() {
            let operand_id = idx as u32;
            if !operand.descriptor.shape.is_empty()
                || matches!(operand.kind, rustnn::graph::OperandKind::Constant)
            {
                continue;
            }

            let name = operand
                .name
                .clone()
                .unwrap_or_else(|| format!("operand_{}", operand_id));
            if let Some((op_type, inputs)) = producer.get(&operand_id) {
                let mut input_descs = Vec::with_capacity(inputs.len());
                for input_id in inputs {
                    let input_op = &self.graph_info.operands[*input_id as usize];
                    let input_name = input_op
                        .name
                        .clone()
                        .unwrap_or_else(|| format!("operand_{}", input_id));
                    input_descs.push(format!("{}{:?}", input_name, input_op.descriptor.shape));
                }
                out.push(format!(
                    "{} (id={}, op={}, inputs=[{}])",
                    name,
                    operand_id,
                    op_type,
                    input_descs.join(", ")
                ));
            } else {
                out.push(format!("{} (id={}, op=<none>)", name, operand_id));
            }
        }

        out
    }

    /// Debug unknown shapes as structured data.
    fn debug_unknown_shapes_structured<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        use pyo3::types::PyDict;
        use std::collections::HashMap;

        let mut producer: HashMap<u32, (String, Vec<u32>)> = HashMap::new();
        for op in &self.graph_info.operations {
            let op_type = op.op_type().to_string();
            let input_ids = op.input_operands();
            for &output_id in op.output_operands_slice() {
                producer.insert(output_id, (op_type.clone(), input_ids.clone()));
            }
        }

        let mut out: Vec<Bound<PyAny>> = Vec::new();
        for (idx, operand) in self.graph_info.operands.iter().enumerate() {
            let operand_id = idx as u32;
            if !operand.descriptor.shape.is_empty()
                || matches!(operand.kind, rustnn::graph::OperandKind::Constant)
            {
                continue;
            }

            let name = operand
                .name
                .clone()
                .unwrap_or_else(|| format!("operand_{}", operand_id));
            if let Some((op_type, inputs)) = producer.get(&operand_id) {
                let mut input_descs: Vec<Bound<PyAny>> = Vec::with_capacity(inputs.len());
                for input_id in inputs {
                    let input_op = &self.graph_info.operands[*input_id as usize];
                    let input_name = input_op
                        .name
                        .clone()
                        .unwrap_or_else(|| format!("operand_{}", input_id));
                    let entry = PyDict::new(py);
                    entry.set_item("id", *input_id)?;
                    entry.set_item("name", input_name)?;
                    entry.set_item("shape", input_op.descriptor.static_or_max_shape())?;
                    input_descs.push(entry.into_any());
                }
                let entry = PyDict::new(py);
                entry.set_item("id", operand_id)?;
                entry.set_item("name", name)?;
                entry.set_item("op", op_type)?;
                entry.set_item("inputs", input_descs)?;
                out.push(entry.into_any());
            } else {
                let entry = PyDict::new(py);
                entry.set_item("id", operand_id)?;
                entry.set_item("name", name)?;
                entry.set_item("op", py.None())?;
                entry.set_item("inputs", Vec::<Bound<PyAny>>::new())?;
                out.push(entry.into_any());
            }
        }

        Ok(out)
    }

    /// Save the graph to a .webnn JSON file
    ///
    /// Args:
    ///     path: File path to save the graph (e.g., "model.webnn")
    ///     quantized: When True, mark the serialized graph as quantized in the header
    ///
    /// Example:
    ///     graph.save("my_model.webnn")
    #[pyo3(signature = (path, quantized=false))]
    fn save(&self, path: &str, quantized: bool) -> PyResult<()> {
        // Convert GraphInfo to GraphJson
        let graph_json = webnn_json::to_graph_json(&self.graph_info, quantized)
            .map_err(|e| PyIOError::new_err(format!("Failed to convert graph: {}", e)))?;

        // Serialize to JSON
        let json_string = serde_json::to_string_pretty(&graph_json)
            .map_err(|e| PyIOError::new_err(format!("Failed to serialize to JSON: {}", e)))?;

        // Write to file
        fs::write(path, json_string)
            .map_err(|e| PyIOError::new_err(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Load a graph from a .webnn file (JSON or text format)
    ///
    /// Args:
    ///     path: File path to load the graph from (e.g., "model.webnn")
    ///     manifest_path: Optional path to manifest.json for manifest + raw weights layout
    ///     weights_path: Path to a `.safetensors` file, or to a raw `.weights` blob (with manifest
    ///         passed explicitly or discovered next to the graph). Relative paths are resolved from
    ///         the graph file’s parent directory (same rules as `webnn-graph`).
    ///
    /// Returns:
    ///     MLGraph: The loaded graph
    ///
    /// The loader automatically detects the format:
    /// - JSON format: Legacy format with embedded base64 weights
    /// - Text format: WebNN DSL format (automatically detected)
    ///
    /// Example:
    ///     graph = MLGraph.load("my_model.webnn")
    ///     graph = MLGraph.load("model.webnn", manifest_path="manifest.json", weights_path="model.weights")
    ///     graph = MLGraph.load("model.webnn", weights_path="model.safetensors")
    ///     graph = MLGraph.load("model.webnn", weights_path="custom_name.safetensors")
    #[staticmethod]
    #[pyo3(signature = (path, manifest_path=None, weights_path=None))]
    fn load(path: &str, manifest_path: Option<&str>, weights_path: Option<&str>) -> PyResult<Self> {
        // Check if file exists
        let path_obj = Path::new(path);
        if !path_obj.exists() {
            return Err(PyIOError::new_err(format!("File not found: {}", path)));
        }

        // Read file
        let content = fs::read_to_string(path)
            .map_err(|e| PyIOError::new_err(format!("Failed to read file: {}", e)))?;

        // Try to detect format and parse accordingly
        let mut graph_json: webnn_graph::ast::GraphJson = if content.trim().starts_with('{') {
            // JSON format
            serde_json::from_str(&content)
                .map_err(|e| PyIOError::new_err(format!("Failed to parse JSON: {}", e)))?
        } else {
            // WebNN text DSL format - sanitize identifiers first
            let sanitized = rustnn::loader::sanitize_webnn_identifiers(&content);
            webnn_graph::parser::parse_wg_text(&sanitized).map_err(|e| {
                PyIOError::new_err(format!("Failed to parse WebNN text format: {}", e))
            })?
        };

        // Resolve external weight references if present
        Self::resolve_external_weights(&mut graph_json, path_obj, manifest_path, weights_path)?;

        // Convert GraphJson to GraphInfo
        let graph_info = webnn_json::from_graph_json(&graph_json)
            .map_err(|e| PyIOError::new_err(format!("Failed to convert graph: {}", e)))?;

        Ok(PyMLGraph { graph_info })
    }
}

impl PyMLGraph {
    pub fn new(graph_info: GraphInfo) -> Self {
        Self { graph_info }
    }

    /// Delegates to [`webnn_graph::resolve_external_weights`], then surfaces a Python error if any
    /// `@weights` refs remain (e.g. partial safetensors coverage).
    fn resolve_external_weights(
        graph_json: &mut webnn_graph::ast::GraphJson,
        graph_path: &Path,
        manifest_path: Option<&str>,
        weights_path: Option<&str>,
    ) -> PyResult<()> {
        use webnn_graph::ast::ConstInit;

        webnn_graph::resolve_external_weights(
            graph_json,
            graph_path,
            weights_path,
            manifest_path,
        )
        .map_err(|e| {
            PyIOError::new_err(format!(
                "Failed to resolve external weights: {e}. \
                 Pass weights_path to a `.safetensors` file, or manifest_path and weights_path for manifest + raw blob, \
                 or place sidecar files next to the graph."
            ))
        })?;

        let pending_count = graph_json
            .consts
            .values()
            .filter(|c| matches!(c.init, ConstInit::Weights { .. }))
            .count();
        if pending_count > 0 {
            return Err(PyIOError::new_err(format!(
                "Graph still has {pending_count} external weight reference(s) after resolution. \
                 Pass weights_path to your `.safetensors` file, or manifest_path + weights_path for manifest + raw blob, \
                 or place model.safetensors / {{stem}}.safetensors or manifest + weights next to the graph. \
                 Rebuild the native extension (`pip install -e .` / maturin) if dependency changes are not picked up.",
            )));
        }

        Ok(())
    }
}
