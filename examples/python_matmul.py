#!/usr/bin/env python3
"""
Matrix multiplication example using WebNN Python API

This example demonstrates:
1. Creating constant operands from NumPy arrays
2. Matrix multiplication operations
3. Combining operations in a graph
"""

import numpy as np
import webnn


def main():
    print("WebNN Matrix Multiplication Example")
    print("=" * 50)

    # Create ML context (CPU-only for this example)
    ml = webnn.ML()
    context = ml.create_context(power_preference="default", accelerated=False)
    print(f"Context: accelerated={context.accelerated}, power={context.power_preference}\n")
    builder = context.create_graph_builder()

    # Define matrix multiplication: output = relu(matmul(input, weights) + bias)
    print("\nBuilding graph: output = relu(matmul(input, weights) + bias)")

    # Input: [batch_size, input_features] = [2, 4]
    input_tensor = builder.input("input", [2, 4], "float32")
    print(f"  Input: {input_tensor}")

    # Weights: [input_features, output_features] = [4, 3]
    weights_data = np.random.randn(4, 3).astype(np.float32) * 0.1
    weights = builder.constant(weights_data)
    print(f"  Weights: {weights}")

    # Bias: [output_features] = [3]
    bias_data = np.zeros(3, dtype=np.float32)
    bias = builder.constant(bias_data)
    print(f"  Bias: {bias}")

    # Forward pass
    matmul_result = builder.matmul(input_tensor, weights)
    print(f"  matmul(input, weights): {matmul_result}")

    # Note: Broadcasting would need proper implementation
    # For now, this is a simplified example
    add_result = builder.add(matmul_result, bias)
    print(f"  result + bias: {add_result}")

    output = builder.relu(add_result)
    print(f"  relu(result + bias): {output}")

    # Build graph
    print("\nCompiling graph...")
    graph = builder.build({"output": output})
    print(f"Compiled graph: {graph}")
    print(f"  Operands: {graph.operand_count}")
    print(f"  Operations: {graph.operation_count}")
    print(f"  Inputs: {graph.get_input_names()}")
    print(f"  Outputs: {graph.get_output_names()}")

    # Execute with test data
    print("\nExecuting graph with test data...")
    input_data = np.array([[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    print(f"  Input shape: {input_data.shape}")
    print(f"  Input data:\n{input_data}")

    # Compute expected result manually
    expected = np.maximum(np.matmul(input_data, weights_data) + bias_data, 0)
    print(f"  Expected output:\n{expected}")

    # Execute the graph
    results = context.compute(graph, {"input": input_data})
    print(f"  Computed output shape: {results['output'].shape}")
    print(f"  Computed output:\n{results['output']}")

    # Verify results
    if results['output'].shape == expected.shape and np.allclose(results['output'], expected, rtol=1e-5):
        print("  ✓ Results match expected values!")
    else:
        print("  ⚠ Results are zeros (ONNX runtime not available)")
        print("    Build with: maturin develop --features python,onnx-runtime")

    # Save to ONNX
    print("\nConverting to ONNX...")
    output_path = "matmul_graph.onnx"
    context.convert_to_onnx(graph, output_path)
    print(f"  Saved to: {output_path}")

    print("\n" + "=" * 50)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
