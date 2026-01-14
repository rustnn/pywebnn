"""Example demonstrating the dispatch() API with MLTensor

This example shows how to use the explicit tensor management API
with dispatch() for async-style execution following the W3C WebNN
MLTensor Explainer.

Reference: https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md
"""

import numpy as np
import webnn


def main():
    # Create context
    ml = webnn.ML()
    context = ml.create_context(device_type="cpu")

    # Build a simple graph: output = relu(x + y)
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    y = builder.input("y", [2, 3], "float32")
    z = builder.add(x, y)
    output = builder.relu(z)
    graph = builder.build({"output": output})

    # Create input tensors
    x_tensor = context.create_tensor([2, 3], "float32", readable=True, writable=True)
    y_tensor = context.create_tensor([2, 3], "float32", readable=True, writable=True)

    # Create output tensor (must be writable for dispatch to write results)
    output_tensor = context.create_tensor(
        [2, 3], "float32", readable=True, writable=True
    )

    # Write input data
    x_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    y_data = np.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], dtype=np.float32)

    context.write_tensor(x_tensor, x_data)
    context.write_tensor(y_tensor, y_data)

    # Dispatch computation
    context.dispatch(graph, {"x": x_tensor, "y": y_tensor}, {"output": output_tensor})

    # Read output
    result = context.read_tensor(output_tensor)
    print("Input x:")
    print(x_data)
    print("\nInput y:")
    print(y_data)
    print("\nOutput (relu(x + y)):")
    print(result)

    # Verify result
    expected = np.maximum(x_data + y_data, 0)
    assert np.allclose(result, expected), "Result doesn't match expected output"
    print("\nVerification: PASSED")

    # Clean up
    x_tensor.destroy()
    y_tensor.destroy()
    output_tensor.destroy()


if __name__ == "__main__":
    main()
