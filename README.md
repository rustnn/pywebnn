# pywebnn

Python bindings for the [W3C WebNN (Web Neural Network) API](https://www.w3.org/TR/webnn/) specification.

## Overview

`pywebnn` provides Python bindings to the `rustnn` library, enabling Python developers to use the WebNN API for neural network inference across multiple backends:

- **ONNX Runtime** - CPU and GPU inference (cross-platform)
- **TensorRT** - NVIDIA GPU inference (Linux/Windows)
- **CoreML** - Apple Neural Engine and GPU (macOS)

## Relationship to rustnn

`pywebnn` is the Python interface to the [rustnn](https://github.com/rustnn/rustnn) Rust library, which implements the W3C WebNN specification. All core functionality (graph validation, backend execution, model conversion) is implemented in Rust for performance and safety.

**Architecture:**
```
Python Application
       ↓
    pywebnn (PyO3 bindings)
       ↓
    rustnn (Rust core)
       ↓
Backend (ONNX Runtime / TensorRT / CoreML)
```

## Installation

### From PyPI (when published)

```bash
pip install pywebnn
```

### From Source

Requirements:
- Python 3.11 or later
- Rust toolchain (1.70+)
- [maturin](https://github.com/PyO3/maturin)

```bash
# Clone the repository
git clone https://github.com/rustnn/pywebnn.git
cd pywebnn

# Install in development mode
maturin develop

# Or build a wheel
maturin build --release
pip install target/wheels/pywebnn-*.whl
```

## Quick Start

```python
import webnn
import numpy as np

# Create context and graph builder
ml = webnn.ML()
context = ml.create_context(device_type="cpu")
builder = context.create_graph_builder()

# Build a simple graph: output = relu(x + y)
x = builder.input("x", [2, 3], "float32")
y = builder.input("y", [2, 3], "float32")
z = builder.add(x, y)
output = builder.relu(z)

# Compile the graph
graph = builder.build({"output": output})

# Execute the graph
x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
y_data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

results = context.compute(graph, {
    "x": x_data,
    "y": y_data
})

print(results["output"])
# Output: [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]
```

## Features

### W3C WebNN API Compliance

- Full implementation of the W3C WebNN specification
- 88 operations supported (84% spec coverage)
- Async execution with `MLTensor` and `dispatch()`
- Backend selection using context hints (power_preference, accelerated)

### Multiple Backends

- **ONNX Runtime**: CPU and GPU execution (automatically selects best device)
- **TensorRT**: High-performance NVIDIA GPU execution
- **CoreML**: Apple Silicon Neural Engine and GPU

### Python Integration

- NumPy array interface
- Type stubs for IDE autocomplete
- Async/await support via `AsyncMLContext`
- Model loading from Hugging Face Hub

## API Classes

- `ML` - Entry point for creating contexts
- `MLContext` - Execution context with backend selection
- `MLGraphBuilder` - Construct neural network graphs
- `MLGraph` - Compiled, executable graph
- `MLOperand` - Graph nodes (tensors)
- `MLTensor` - Explicit tensor management with device memory
- `AsyncMLContext` - Async wrapper for non-blocking execution

## Examples

See the [examples/](examples/) directory for complete working examples:

- `python_simple.py` - Basic graph construction
- `python_matmul.py` - Matrix multiplication
- `image_classification.py` - MobileNetV2 inference
- `text_generation_gpt.py` - GPT-style text generation
- And more...

## Documentation

- [API Reference](docs/api-reference.md) - Complete API documentation
- [Examples Guide](docs/examples.md) - Detailed usage examples
- [Getting Started](docs/getting-started.md) - Installation and first steps
- [Advanced Topics](docs/advanced.md) - Performance tuning, custom backends

## Development

### Building from Source

```bash
# Install development dependencies
pip install maturin pytest pytest-asyncio

# Build and install in development mode
maturin develop

# Run tests
pytest tests/ -v
```

### Contributing

Contributions are welcome! Please see the main [rustnn](https://github.com/rustnn/rustnn) repository for development guidelines and architecture documentation.

## License

Apache-2.0 - See [LICENSE](LICENSE) for details.

## Related Projects

- [rustnn](https://github.com/rustnn/rustnn) - Rust core library
- [webnn-graph](https://github.com/rustnn/webnn-graph) - Graph data structures
- [webnn-onnx-utils](https://github.com/rustnn/webnn-onnx-utils) - ONNX/WebNN utilities

## Links

- [GitHub Repository](https://github.com/rustnn/pywebnn)
- [PyPI Package](https://pypi.org/project/pywebnn/) (when published)
- [Issue Tracker](https://github.com/rustnn/pywebnn/issues)
- [W3C WebNN Specification](https://www.w3.org/TR/webnn/)
