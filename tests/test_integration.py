"""Native onnx context integration with persistent tensors."""

import numpy as np
import webnn
from runtime_support import requires_execution_backend


@requires_execution_backend
def test_native_backend_dispatch():
    context = webnn.ML().create_context(backend="onnx", accelerated=False)
    builder = context.create_graph_builder()
    x = builder.input("x", [2], "float32")
    graph = builder.build({"y": builder.add(x, x)})
    source = context.create_tensor([2], "float32", writable=True)
    target = context.create_tensor([2], "float32", readable=True)
    context.write_tensor(source, np.array([1, 2], dtype=np.float32))
    context.dispatch(graph, {"x": source}, {"y": target})
    np.testing.assert_array_equal(context.read_tensor(target), [2, 4])
