"""Repeated dispatch reuses explicit tensor bindings."""

import numpy as np
import pytest
from runtime_support import requires_execution_backend


@requires_execution_backend
def test_repeated_dispatch_reuses_tensors(context):
    builder = context.create_graph_builder()
    x = builder.input("x", [1, 3, 32, 32], "float32")
    y = x
    for _ in range(10):
        y = builder.relu(builder.add(y, x))
    graph = builder.build({"y": y})
    source = context.create_tensor([1, 3, 32, 32], "float32", writable=True)
    target = context.create_tensor([1, 3, 32, 32], "float32", readable=True)
    values = np.ones([1, 3, 32, 32], dtype=np.float32)
    context.write_tensor(source, values)
    inputs, outputs = {"x": source}, {"y": target}
    for _ in range(20):
        context.dispatch(graph, inputs, outputs)
    np.testing.assert_array_equal(context.read_tensor(target), 11 * values)
