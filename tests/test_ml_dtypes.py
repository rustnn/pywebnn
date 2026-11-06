"""Low precision NumPy conversions and native backend boundaries."""

import subprocess
import sys

import ml_dtypes
import numpy as np
import pytest
import webnn

DTYPES = [
    pytest.param("bfloat16", ml_dtypes.bfloat16, id="bfloat16"),
    pytest.param("float8e4m3", ml_dtypes.float8_e4m3fn, id="float8e4m3"),
    pytest.param("float8_e4m3fn", ml_dtypes.float8_e4m3fn, id="numpy_float8_alias"),
]


@pytest.fixture
def onnx_context():
    return webnn.ML().create_context(backend="onnx", accelerated=False)


@pytest.mark.parametrize("name,dtype", DTYPES)
@pytest.mark.parametrize("layout", ["typed", "float32", "strided", "scalar"])
def test_tensor_conversion(onnx_context, name, dtype, layout):
    values = np.array(
        [[0.0, -0.0, 1.1, -2.3], [0.125, 3.5, 7.75, np.nan]], dtype=np.float32
    )
    if layout == "typed":
        values = values.astype(dtype)
    elif layout == "strided":
        values = values[:, ::-1]
        assert not values.flags.c_contiguous
    elif layout == "scalar":
        values = np.array(-1.1, dtype=np.float32)
    expected = values.astype(dtype)
    tensor = onnx_context.create_tensor(
        list(values.shape), name, readable=True, writable=True
    )
    onnx_context.write_tensor(tensor, values)
    result = onnx_context.read_tensor(tensor)
    assert result.dtype == np.dtype(dtype)
    assert result.shape == values.shape
    # Compare storage to catch float16/bfloat16 confusion and preserve signed zero/NaN.
    assert result.tobytes() == expected.tobytes()


@pytest.mark.parametrize("name,dtype", DTYPES)
def test_all_bit_patterns_roundtrip(onnx_context, name, dtype):
    carrier = np.uint16 if name == "bfloat16" else np.uint8
    values = np.arange(1 << (8 * np.dtype(carrier).itemsize), dtype=carrier).view(dtype)
    tensor = onnx_context.create_tensor(
        list(values.shape), name, readable=True, writable=True
    )
    onnx_context.write_tensor(tensor, values)
    assert onnx_context.read_tensor(tensor).tobytes() == values.tobytes()


@pytest.mark.parametrize("name,dtype", DTYPES)
@pytest.mark.parametrize("infer", [True, False])
def test_constant_dtype(onnx_context, name, dtype, infer):
    builder = onnx_context.create_graph_builder()
    values = np.array([[1.0, -2.5], [0.125, 3.0]], dtype=dtype if infer else np.float32)
    operand = (
        builder.constant(values) if infer else builder.constant(values, data_type=name)
    )
    assert builder.operand_shape(operand) == [2, 2]
    assert builder.operand_data_type(operand) == (
        "bfloat16" if name == "bfloat16" else "float8e4m3"
    )


@pytest.mark.parametrize("name,dtype", DTYPES)
def test_onnx_execution_limit(onnx_context, name, dtype):
    builder = onnx_context.create_graph_builder()
    source = builder.input("x", [2], name)
    result = builder.cast(source, "float32")
    with pytest.raises(RuntimeError, match="Backend Onnx does not support data type"):
        builder.build({"y": result})


@pytest.mark.parametrize("dtype", [ml_dtypes.float8_e5m2, ml_dtypes.float8_e4m3fnuz])
def test_unsupported_float8_format(onnx_context, dtype):
    builder = onnx_context.create_graph_builder()
    with pytest.raises(ValueError, match="Unsupported data type"):
        builder.constant(np.array([1.0], dtype=dtype))


def test_conversion_imports_ml_dtypes():
    # A fresh process has not imported ml_dtypes on behalf of the binding.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import webnn
assert 'ml_dtypes' not in sys.modules
context = webnn.ML().create_context(backend='onnx', accelerated=False)
for name in ('bfloat16', 'float8e4m3'):
    tensor = context.create_tensor([2], name, readable=True, writable=True)
    context.write_tensor(tensor, [1.0, -2.5])
    result = context.read_tensor(tensor)
    assert result.astype('float32').tolist() == [1.0, -2.5]
""",
        ],
        check=True,
        timeout=60,
    )
