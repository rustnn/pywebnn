"""Run the same graph through the TensorRT RTX ORT plugin and native trtx.

From a pywebnn checkout, with a virtual environment active:
    MATURIN_PEP517_ARGS="--features onnx-runtime,trtx-runtime" \
        pip install -e '.[ort-trtx,trtx]'
    python examples/ort_trtx.py

Requires an NVIDIA RTX GPU and a compatible CUDA 13 driver.
"""

import numpy as np
import webnn


def run_graph(context: webnn.MLContext, values: np.ndarray) -> np.ndarray:
    """Build and dispatch relu(x + x) using the supplied context."""
    if not context.accelerated:
        raise RuntimeError("An accelerated context is required")
    builder = context.create_graph_builder()
    x = builder.input("x", list(values.shape), "float32")
    y = builder.relu(builder.add(x, x))
    graph = builder.build({"y": y})
    source = context.create_tensor(list(values.shape), "float32", writable=True)
    target = context.create_tensor(list(values.shape), "float32", readable=True)
    context.write_tensor(source, values)
    context.dispatch(graph, {"x": source}, {"y": target})
    return context.read_tensor(target).copy()


def main() -> None:
    import onnxruntime_ep_nv_tensorrt_rtx as ep

    print(f"EP library path: {ep.get_library_path()}")
    # Register the ORT plugin before creating either context.
    webnn.init_global_ort_env(
        [(ep.get_ep_name(), ep.get_library_path())], log_level="info"
    )
    values = np.arange(-3, 3, dtype=np.float32).reshape(2, 3)
    expected = np.maximum(values * 2, 0)

    ml = webnn.ML()
    ort_context = ml.create_context(backend="onnx", accelerated=True)
    trtx_context = ml.create_context(backend="trtx", accelerated=True)
    ort_result = run_graph(ort_context, values)
    trtx_result = run_graph(trtx_context, values)
    np.testing.assert_array_equal(ort_result, expected)
    np.testing.assert_array_equal(trtx_result, expected)
    np.testing.assert_array_equal(ort_result, trtx_result)

    print(f"ORT plugin ({ep.get_ep_name()}):\n{ort_result}")
    print(f"Native trtx:\n{trtx_result}")
    print("Both GPU backends match NumPy and each other.")


if __name__ == "__main__":
    main()
