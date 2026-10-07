"""Native coreml context integration with persistent tensors."""

import itertools

import numpy as np
import pytest
import webnn
from runtime_support import (
    AVAILABLE_BACKEND_ACCELERATED_COMBOS,
    TESTED_BACKENDS,
)


@pytest.mark.parametrize(
    ("backend", "accelerated"), list(itertools.product(TESTED_BACKENDS, (True, False)))
)
def test_native_backend_dispatch(backend, accelerated):
    if (backend, accelerated) not in AVAILABLE_BACKEND_ACCELERATED_COMBOS:
        pytest.skip(f"{backend} with {accelerated=} not available")

    context = webnn.ML().create_context(backend=backend, accelerated=accelerated)
    builder = context.create_graph_builder()
    x = builder.input("x", [2], "float32")
    graph = builder.build({"y": builder.add(x, x)})
    source = context.create_tensor([2], "float32", writable=True)
    target = context.create_tensor([2], "float32", readable=True)
    context.write_tensor(source, np.array([1, 2], dtype=np.float32))
    context.dispatch(graph, {"x": source}, {"y": target})
    np.testing.assert_array_equal(context.read_tensor(target), [2, 4])
