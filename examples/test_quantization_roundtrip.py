"""Save, reload, and execute a native quantized graph with explicit tensors."""

import tempfile
from pathlib import Path
import numpy as np
import webnn


def test_quantization_roundtrip(quant_dtype="int8", quant_level_name="int8"):
    context = webnn.ML().create_context(accelerated=False)
    builder = context.create_graph_builder()
    x = builder.input("x", [4], "float32")
    scale = builder.constant(np.array(0.5, dtype=np.float32))
    zero = builder.constant(
        np.array(0, dtype=np.int8 if quant_dtype.startswith("int") else np.uint8),
        data_type=quant_dtype,
    )
    y = builder.dequantize_linear(builder.quantize_linear(x, scale, zero), scale, zero)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "quantized.webnn"
        builder.save({"y": y}, str(path))
        assert path.exists() and path.with_suffix(".safetensors").exists()
        graph = builder.build({"y": y})
        source = context.create_tensor([4], "float32", writable=True)
        target = context.create_tensor([4], "float32", readable=True)
        values = np.arange(4, dtype=np.float32) * 0.5
        context.write_tensor(source, values)
        context.dispatch(graph, {"x": source}, {"y": target})
        np.testing.assert_allclose(context.read_tensor(target), values)
        loaded = context.load_graph(str(path))
        context.dispatch(loaded, {"x": source}, {"y": target})
        np.testing.assert_allclose(context.read_tensor(target), values)
    print(f"{quant_level_name}: native save/load round trip and dispatch passed")
    return True


if __name__ == "__main__":
    for dtype in ["int8", "uint8", "int4", "uint4"]:
        test_quantization_roundtrip(dtype, dtype)
