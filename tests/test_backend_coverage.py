"""Execution coverage for explicit CPU/ONNX and CoreML backends."""

import numpy as np
import pytest

from runtime_support import COREML_BACKEND_AVAILABLE, EXECUTION_BACKEND_AVAILABLE

try:
    import webnn
except ImportError:
    pytestmark = pytest.mark.skip(reason="webnn not built yet")


def _context_for_backend(backend_name):
    ml = webnn.ML()
    if backend_name == "cpu":
        return ml.create_context(
            power_preference="default",
            accelerated=False,
            device_type="cpu",
            backend="onnx",
        )
    if backend_name == "coreml":
        return ml.create_context(
            power_preference="default",
            accelerated=True,
            device_type="npu",
            backend="coreml",
        )
    raise ValueError(f"unsupported backend: {backend_name}")


def _skip_if_unavailable(backend_name):
    if backend_name == "cpu" and not EXECUTION_BACKEND_AVAILABLE:
        pytest.skip("No working CPU execution backend is available")
    if backend_name == "coreml" and not COREML_BACKEND_AVAILABLE:
        pytest.skip("No working CoreML execution backend is available")


@pytest.mark.parametrize("backend_name", ["cpu", "coreml"])
def test_add_relu_execute_on_backend(backend_name):
    """Basic numeric graph executes on both backend families."""
    _skip_if_unavailable(backend_name)
    context = _context_for_backend(backend_name)
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 2], "float32")
    y = builder.input("y", [2, 2], "float32")
    output = builder.relu(builder.add(x, y))
    graph = builder.build({"output": output})

    x_data = np.array([[1.0, -4.0], [2.0, -3.0]], dtype=np.float32)
    y_data = np.array([[2.0, 1.0], [-5.0, 5.0]], dtype=np.float32)
    result = context.compute(graph, {"x": x_data, "y": y_data})

    np.testing.assert_allclose(result["output"], np.maximum(x_data + y_data, 0.0))


@pytest.mark.parametrize("backend_name", ["cpu", "coreml"])
def test_comparison_execute_on_backend(backend_name):
    """Comparison outputs execute and round-trip as WebNN uint8 booleans."""
    _skip_if_unavailable(backend_name)
    context = _context_for_backend(backend_name)
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 2], "float32")
    y = builder.input("y", [2, 2], "float32")
    output = builder.greater(x, y)
    graph = builder.build({"output": output})

    x_data = np.array([[1.0, 4.0], [2.0, 3.0]], dtype=np.float32)
    y_data = np.array([[2.0, 1.0], [2.0, 5.0]], dtype=np.float32)
    result = context.compute(graph, {"x": x_data, "y": y_data})

    np.testing.assert_array_equal(result["output"], (x_data > y_data).astype(np.uint8))


@pytest.mark.parametrize("backend_name", ["cpu", "coreml"])
def test_quantize_linear_with_constant_zero_point_execute_on_backend(backend_name):
    """Quantization executes on both backends when CoreML-required constants are used."""
    _skip_if_unavailable(backend_name)
    context = _context_for_backend(backend_name)
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 2], "float32")
    scale = builder.constant(np.array([[0.5]], dtype=np.float32))
    zero_point = builder.constant(np.array([[0]], dtype=np.int8))
    output = builder.quantize_linear(x, scale, zero_point)
    graph = builder.build({"output": output})

    x_data = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=np.float32)
    result = context.compute(graph, {"x": x_data})

    np.testing.assert_array_equal(result["output"], np.array([[0, 2], [4, -2]], dtype=np.int8))


@pytest.mark.xfail(
    strict=True,
    reason="CoreML logical ops currently require bool inputs; WebNN numeric truthiness is not lowered yet.",
)
def test_coreml_logical_numeric_inputs():
    _skip_if_unavailable("coreml")
    context = _context_for_backend("coreml")
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 2], "float32")
    y = builder.input("y", [2, 2], "float32")
    graph = builder.build({"output": builder.logical_and(x, y)})

    context.compute(
        graph,
        {
            "x": np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32),
            "y": np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        },
    )


@pytest.mark.xfail(
    strict=True,
    reason="CoreML comparison-to-logical chains currently emit a duplicate bool temporary name.",
)
def test_coreml_logical_comparison_inputs():
    _skip_if_unavailable("coreml")
    context = _context_for_backend("coreml")
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 2], "float32")
    y = builder.input("y", [2, 2], "float32")
    z = builder.input("z", [2, 2], "float32")
    lhs = builder.greater(x, y)
    rhs = builder.lesser(x, z)
    graph = builder.build({"output": builder.logical_and(lhs, rhs)})

    context.compute(
        graph,
        {
            "x": np.array([[1.0, 4.0], [2.0, 3.0]], dtype=np.float32),
            "y": np.ones((2, 2), dtype=np.float32),
            "z": np.full((2, 2), 3.0, dtype=np.float32),
        },
    )


@pytest.mark.xfail(
    strict=True,
    reason="CoreML where/select currently requires bool conditions; WebNN integer truthiness is not lowered yet.",
)
def test_coreml_where_integer_condition():
    _skip_if_unavailable("coreml")
    context = _context_for_backend("coreml")
    builder = context.create_graph_builder()
    condition = builder.input("condition", [2, 2], "int32")
    true_value = builder.input("true_value", [2, 2], "float32")
    false_value = builder.input("false_value", [2, 2], "float32")
    graph = builder.build({"output": builder.where_(condition, true_value, false_value)})

    context.compute(
        graph,
        {
            "condition": np.array([[0, 1], [1, 0]], dtype=np.int32),
            "true_value": np.ones((2, 2), dtype=np.float32),
            "false_value": np.zeros((2, 2), dtype=np.float32),
        },
    )


@pytest.mark.xfail(
    strict=True,
    reason="CoreML quantize/dequantize currently require zero_point to be constant.",
)
def test_coreml_quantize_linear_dynamic_zero_point():
    _skip_if_unavailable("coreml")
    context = _context_for_backend("coreml")
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 2], "float32")
    scale = builder.input("scale", [1, 1], "float32")
    zero_point = builder.input("zero_point", [1, 1], "int8")
    graph = builder.build({"output": builder.quantize_linear(x, scale, zero_point)})

    context.compute(
        graph,
        {
            "x": np.array([[0.0, 1.0], [2.0, -1.0]], dtype=np.float32),
            "scale": np.array([[0.5]], dtype=np.float32),
            "zero_point": np.array([[0]], dtype=np.int8),
        },
    )
