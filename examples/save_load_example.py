"""Save and reload through RustNN, then dispatch with explicit tensors."""

from pathlib import Path
import numpy as np
import webnn


def main():
    context = webnn.ML().create_context(accelerated=False)
    builder = context.create_graph_builder()
    x = builder.input("x", [2], "float32")
    y = builder.add(x, builder.constant(np.ones(2, dtype=np.float32)))
    outputs = {"y": y}
    builder.save(outputs, "example_graph.webnn")
    graph = builder.build(outputs)
    source = context.create_tensor([2], "float32", writable=True)
    target = context.create_tensor([2], "float32", readable=True)
    context.write_tensor(source, np.array([1, 2], dtype=np.float32))
    context.dispatch(graph, {"x": source}, {"y": target})
    expected = context.read_tensor(target)
    loaded = context.load_graph("example_graph.webnn")
    context.dispatch(loaded, {"x": source}, {"y": target})
    np.testing.assert_array_equal(context.read_tensor(target), expected)
    print(expected)
    print(Path("example_graph.webnn"), Path("example_graph.safetensors"))


if __name__ == "__main__":
    main()
