"""Runtime capability checks and pytest markers shared by the test suite."""

import platform

import numpy as np
import pytest


def _can_execute(*, device_type: str, accelerated: bool) -> bool:
    """Return whether a minimal graph can execute on the requested device."""
    try:
        import webnn

        ml = webnn.ML()
        context = ml.create_context(
            power_preference="default",
            accelerated=accelerated,
            device_type=device_type,
        )
        builder = context.create_graph_builder()
        input_operand = builder.input("x", [1, 1], "float32")
        graph = builder.build({"output": builder.relu(input_operand)})
        result = context.compute(graph, {"x": np.array([[1.0]], dtype=np.float32)})
        return bool(np.any(result["output"] != 0))
    except Exception:
        return False


EXECUTION_BACKEND_AVAILABLE = _can_execute(device_type="cpu", accelerated=False)
COREML_BACKEND_AVAILABLE = platform.system() == "Darwin" and _can_execute(
    device_type="npu", accelerated=True
)

requires_execution_backend = pytest.mark.skipif(
    not EXECUTION_BACKEND_AVAILABLE,
    reason="No working CPU execution backend is available",
)
requires_coreml_backend = pytest.mark.skipif(
    not COREML_BACKEND_AVAILABLE,
    reason="No working CoreML execution backend is available",
)
