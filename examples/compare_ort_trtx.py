"""Compare the TensorRT RTX ORT plugin with the native TensorRT RTX backend.

From a pywebnn checkout, with a virtual environment active:
    MATURIN_PEP517_ARGS="--features onnx-runtime,trtx-runtime" \
        pip install -e '.[ort-trtx,trtx]'
    python examples/compare_ort_trtx.py

Requires an NVIDIA RTX GPU and a compatible CUDA 13 driver.
"""

import numpy as np
import onnxruntime_ep_nv_tensorrt_rtx as ort_trt
import webnn


def run_graph(backend: str, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Build and execute relu(values @ weights) on the requested backend."""
    context = webnn.ML().create_context(backend=backend, accelerated=True)
    if not context.accelerated:
        raise RuntimeError(f"{backend} did not select an accelerated device")

    builder = context.create_graph_builder()
    x = builder.input("x", list(values.shape), "float32")
    y = builder.relu(builder.matmul(x, builder.constant(weights)))
    output_shape = builder.operand_shape(y)
    graph = builder.build({"y": y})

    source = context.create_tensor(list(values.shape), "float32", writable=True)
    target = context.create_tensor(output_shape, "float32", readable=True)
    context.write_tensor(source, values)
    context.dispatch(graph, {"x": source}, {"y": target})
    # Own the result independently of the context and its tensors.
    return context.read_tensor(target).copy()


def main() -> None:
    # Install the plugin environment before either backend creates a context.
    webnn.init_global_ort_env([(ort_trt.get_ep_name(), ort_trt.get_library_path())])

    values = np.array([[-3, -2, -1], [0, 1, 2]], dtype=np.float32)
    weights = np.array([[1, -1], [2, 1], [-1, 2]], dtype=np.float32)
    expected = np.maximum(values @ weights, 0)

    ort_result = run_graph("onnx", values, weights)
    trtx_result = run_graph("trtx", values, weights)
    np.testing.assert_allclose(ort_result, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(trtx_result, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ort_result, trtx_result, rtol=1e-5, atol=1e-6)

    print(f"ORT plugin ({ort_trt.get_ep_name()}):\n{ort_result}")
    print(f"Native trtx:\n{trtx_result}")
    print("Both GPU backends match NumPy and each other.")


if __name__ == "__main__":
    main()
