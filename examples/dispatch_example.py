"""Build and dispatch through RustNN using persistent tensors."""

import numpy as np
import webnn


def main():
    context = webnn.ML().create_context(accelerated=False, backend="onnx")
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    y = builder.relu(builder.add(x, x))
    output_shape = builder.operand_shape(y)
    graph = builder.build({"y": y})
    source = context.create_tensor([2, 3], "float32", writable=True)
    target = context.create_tensor(output_shape, "float32", readable=True)
    context.write_tensor(source, np.arange(-3, 3, dtype=np.float32).reshape(2, 3))
    inputs, outputs = {"x": source}, {"y": target}
    for _ in range(20):
        context.dispatch(graph, inputs, outputs)
    print(context.read_tensor(target))


if __name__ == "__main__":
    main()
