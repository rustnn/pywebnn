# pywebnn

Python bindings to RustNN's graph builder and persistent tensor API.
RustNN owns graph recording, validation, compilation, execution, and saving.
Python converts arguments and NumPy buffers and retains native object owners.

## Install and develop

```sh
pip install pywebnn
# From a checkout, with a Python virtual environment active:
pip install maturin numpy pytest
maturin develop
```

Install every optional dependency group (including tests and both CUDA variants
of the ORT plugin) with `pip install 'pywebnn[all]'`, or `pip install -e '.[all]'`
from a checkout. For a specific CUDA runtime, select `ort-trtx` (CUDA 13) or
`ort-trtx-cu12` (CUDA 12) individually. Python extras install dependencies;
Rust backend features still need to be enabled when building.

Select a compiled RustNN backend with `backend="onnx"`, `"coreml"`, `"trtx"`,
`"litert"`, or `"cann"`; `"auto"` uses RustNN's selection. Extra runtime features
must be enabled when building, for example `maturin develop --features trtx-runtime`.
Provide the runtime libraries required by that backend.

To install a local checkout with features enabled you can use for example

```bash
# install with coreml-runtime feature
MATURIN_PEP517_ARGS="--features coreml-runtime" pip install -e '.'
# install with trtx-runtime-runtime feature and TensorRT libraries via pip
MATURIN_PEP517_ARGS="--features trtx-runtime" pip install -e '.[trtx]'
```

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

## Export and load

```python
outputs = {"y": y}
builder.save(outputs, "model.webnn")
graph = builder.build(outputs)
```

Saving forwards to RustNN's `rustnn_save_webnn()` and writes `.webnn` text plus
`.safetensors` weights. Reload with `graph = context.load_graph("model.webnn")`,
which delegates loading and compilation to RustNN. The graph's native
`input_descriptors` and `output_descriptors` describe the tensors to allocate.

Export without initializing a runtime using `webnn.MLGraphBuilder.new_uncompiled()`.
Its `build(outputs)` returns an in-memory graph that compiles on its first
`dispatch()` in a context. Read-only `operand.shape` and
`operand.data_type` expose RustNN's inferred metadata.

## Documentation

- [Getting started](docs/getting-started.md)
- [Python API](docs/api-reference.md)
- [Examples](docs/examples.md)
- [RustNN](https://github.com/rustnn/rustnn)

## ONNX Runtime EP plugins

Install the TensorRT RTX ONNX Runtime EP with `pip install 'pywebnn[ort-trtx]'`
(CUDA 13), or `pip install 'pywebnn[ort-trtx-cu12]'` for CUDA 12. See NVIDIA's
[installation requirements](https://nvidia.github.io/TensorRT-RTX-EP-ABI/installation.html).
These extras use the ONNX backend; the `trtx` extra is for the separate native backend.

Register EP libraries before creating a context or discovering devices:

```python
import webnn
import onnxruntime_ep_nv_tensorrt_rtx as trt_ep

webnn.init_global_ort_env(
    [(trt_ep.get_ep_name(), trt_ep.get_library_path())], log_level="warning"
)
context = webnn.ML().create_context(backend="onnx", accelerated=True)
```

Pass a list of `(registration_name, library_path)` pairs to load other EP plugins.
Paths accept strings or `pathlib.Path`. The keyword-only `log_level` accepts
`"verbose"`, `"info"`, `"warning"`, `"error"`, or `"fatal"` (case-insensitive).
It sets the ORT environment's severity threshold before plugin registration;
`None` (the default) preserves ORT's logging configuration. Invalid levels raise
`ValueError` before runtime initialization. Library loading errors raise `RuntimeError`;
a second successful initialization or initialization after context creation raises
`webnn.OrtAlreadyInitialized`. Registration must succeed for every library before
rustnn receives the environment. If registration fails, retry initialization before
creating contexts; the pending environment retains any plugins already registered. `ORT_DYLIB_PATH` selects the ORT runtime library;
otherwise pywebnn discovers the installed `onnxruntime` wheel as usual.

### Compare the ORT plugin and native TensorRT RTX backends

From a checkout with a virtual environment active, enable both Rust backend
features and install both CUDA 13 runtime extras:

```sh
MATURIN_PEP517_ARGS="--features onnx-runtime,trtx-runtime" pip install -e '.[ort-trtx,trtx]'
python examples/compare_ort_trtx.py
```

[`examples/compare_ort_trtx.py`](examples/compare_ort_trtx.py) registers the ORT
plugin before creating contexts, then builds and dispatches the same matrix
multiplication and ReLU graph using `backend="onnx"` and `backend="trtx"`.
It checks both GPU outputs against NumPy and against each other. The example
requires an NVIDIA RTX GPU and a compatible CUDA 13 driver.
