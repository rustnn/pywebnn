"""Runtime probes shared by native binding tests."""

import platform
import numpy as np
import pytest


def _can_execute(backend, accelerated):
    try:
        import webnn

        context = webnn.ML().create_context(backend=backend, accelerated=accelerated)
        builder = context.create_graph_builder()
        x = builder.input("x", [1], "float32")
        graph = builder.build({"y": builder.relu(x)})
        source = context.create_tensor([1], "float32", writable=True)
        target = context.create_tensor([1], "float32", readable=True)
        context.write_tensor(source, np.array([1], dtype=np.float32))
        context.dispatch(graph, {"x": source}, {"y": target})
        return context.read_tensor(target).item() == 1
    except Exception:
        return False


EXECUTION_BACKEND_AVAILABLE = _can_execute("onnx", False)
COREML_BACKEND_AVAILABLE = platform.system() == "Darwin" and _can_execute(
    "coreml", True
)
requires_execution_backend = pytest.mark.skipif(
    not EXECUTION_BACKEND_AVAILABLE, reason="No working ONNX CPU backend"
)
requires_coreml_backend = pytest.mark.skipif(
    not COREML_BACKEND_AVAILABLE, reason="No working CoreML backend"
)
