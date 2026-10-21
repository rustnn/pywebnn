# pywebnn

Thin Python bindings to RustNN's runtime-bound builder and persistent tensor API.
RustNN owns graph recording, validation, compilation, execution, and saving.
Python converts arguments and NumPy buffers and retains native object owners.

## Install and develop

```sh
pip install pywebnn
# From a checkout, with a Python virtual environment active:
pip install maturin numpy pytest
maturin develop
```

Select a compiled RustNN backend with `backend="onnx"`, `"coreml"`, `"trtx"`,
`"litert"`, or `"cann"`; `"auto"` uses RustNN's selection. Extra runtime features
must be enabled when building, for example `maturin develop --features trtx-runtime`.
Provide the runtime libraries required by that backend.

## Build and execute

```python
import numpy as np
import webnn

context = webnn.ML().create_context(accelerated=False, backend="onnx")
builder = context.create_graph_builder()
x = builder.input("x", [2, 3], "float32")
y = builder.relu(builder.add(x, x))
output_shape = builder.operand_shape(y)
graph = builder.build({"y": y})
source = context.create_tensor([2, 3], "float32", writable=True)
target = context.create_tensor(output_shape, "float32", readable=True)
context.write_tensor(source, np.arange(-3, 3, dtype=np.float32).reshape(2, 3))
context.dispatch(graph, {"x": source}, {"y": target})
print(context.read_tensor(target))
```

Allocate tensor bindings once and reuse them across dispatches. There is no
`compute()` method or automatic per-dispatch tensor allocation.

## Save before compilation

```python
outputs = {"y": y}
builder.save(outputs, "model.webnn")
graph = builder.build(outputs)
```

Saving forwards to RustNN's `rustnn_save_webnn()` and writes `.webnn` text plus
`.safetensors` weights. Reload with `graph = context.load_graph("model.webnn")`,
which delegates loading and compilation to RustNN. The graph's native
`input_descriptors` and `output_descriptors` describe the tensors to allocate.

## Documentation

- [Getting started](docs/getting-started.md)
- [Python API](docs/api-reference.md)
- [Examples](docs/examples.md)
- [RustNN](https://github.com/rustnn/rustnn)
