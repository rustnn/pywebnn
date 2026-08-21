# pywebnn

Python bindings for the W3C WebNN API, powered by `rustnn`.

## Install

`pywebnn` is available on PyPI:

```bash
pip install pywebnn
```

## Development

Install the development dependencies and native extension in editable mode:

```powershell
python -m pip install --upgrade pip
python -m pip install maturin numpy tokenizers
maturin develop
```

Re-run `maturin develop` after changing Rust code. To build with TensorRT RTX
support, use `maturin develop --features trtx-runtime` and ensure the matching
TensorRT runtime DLLs are available on `PATH` before launching Python.

## Docs

Full documentation is published on GitHub Pages:

<https://rustnn.github.io/pywebnn/>

## Quick Start

```python
import numpy as np
import webnn

ml = webnn.ML()
context = ml.create_context(device_type="cpu")
builder = context.create_graph_builder()

x = builder.input("x", [2, 3], "float32")
y = builder.input("y", [2, 3], "float32")
out = builder.relu(builder.add(x, y))
graph = builder.build({"output": out})

result = context.compute(
    graph,
    {
        "x": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        "y": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
    },
)
print(result["output"])
```

## TensorRT backend

TensorRT is opt-in. Build an editable installation with RustNN's TensorRT feature,
then request it when creating the context:

```bash
maturin develop --features trtx-runtime
```

```python
context = webnn.ML().create_context(backend="trtx")
```

This requires a compatible TensorRT installation and GPU. Use
`context.backend_info()` to inspect the requested backend and compiled features.
On Windows, make the matching TensorRT RTX runtime DLLs available through
`PATH` before starting Python. The DLL version must match the TensorRT headers
used when building the extension.
Other opt-in runtime plugins are `litert-runtime` and `cann-runtime`; select them
with `backend="litert"` and `backend="cann"`, respectively.

## Links

- GitHub: <https://github.com/rustnn/pywebnn>
- PyPI: <https://pypi.org/project/pywebnn/>
- Issues: <https://github.com/rustnn/pywebnn/issues>
