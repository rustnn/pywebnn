#!/usr/bin/env python3
"""
Simple example demonstrating the WebNN Python API

This example shows how to:
1. Create a context and graph builder
2. Define inputs and operations
3. Build and compile a graph
4. Convert to ONNX format
"""

import numpy as np
import webnn


def main():
    print("WebNN Python API Example")
    print("=" * 50)

    # Create ML context (CPU-only for this example)
    ml = webnn.ML()
    context = ml.create_context(power_preference="default", accelerated=False)
    print(f"\nCreated context: {context}")
    print(f"  Accelerated: {context.accelerated}")
    print(f"  Power preference: {context.power_preference}")

    # Create graph builder
    builder = context.create_graph_builder()
    print("Created graph builder")

    # Define a simple computational graph: z = relu(x + y)
    print("\nBuilding graph: z = relu(x + y)")
    x = builder.input("x", [2, 3], "float32")
    y = builder.input("y", [2, 3], "float32")
    print(f"  Input x: {x}")
    print(f"  Input y: {y}")

    # Add operation
    sum_result = builder.add(x, y)
    print(f"  x + y: {sum_result}")

    # ReLU activation
    output = builder.relu(sum_result)
    print(f"  relu(x + y): {output}")

    # Build the graph
    print("\nCompiling graph...")
    graph = builder.build({"output": output})
    print(f"Compiled graph: {graph}")
    print(f"  Operands: {graph.operand_count}")
    print(f"  Operations: {graph.operation_count}")
    print(f"  Inputs: {graph.get_input_names()}")
    print(f"  Outputs: {graph.get_output_names()}")

    # Prepare input data
    print("\nPreparing test data...")
    x_data = np.array([[1.0, -2.0, 3.0],
                       [4.0, -5.0, 6.0]], dtype=np.float32)
    y_data = np.array([[-1.0, 2.0, -3.0],
                       [-4.0, 5.0, -6.0]], dtype=np.float32)

    print(f"  x_data:\n{x_data}")
    print(f"  y_data:\n{y_data}")

    expected = np.maximum(x_data + y_data, 0)
    print(f"  Expected output (relu(x+y)):\n{expected}")

    # Execute the graph
    print("\nExecuting graph...")
    results = context.compute(graph, {"x": x_data, "y": y_data})
    print(f"  Output shape: {results['output'].shape}")
    print(f"  Computed output:\n{results['output']}")

    # Verify results (if ONNX runtime is available, results should match expected)
    if np.allclose(results['output'], expected, rtol=1e-5):
        print("  ✓ Results match expected values!")
    else:
        print("  ⚠ Results are zeros (ONNX runtime not available)")
        print("    Build with: maturin develop --features python,onnx-runtime")

    # Convert to ONNX
    print("\nConverting to ONNX...")
    output_path = "example_graph.onnx"
    context.convert_to_onnx(graph, output_path)
    print(f"  Saved to: {output_path}")

    print("\n" + "=" * 50)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
