# pywebnn

Python bindings for the W3C WebNN API, powered by `rustnn`.

## Install

`pywebnn` is available on PyPI:

```bash
pip install pywebnn
```

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

## Links

- GitHub: <https://github.com/rustnn/pywebnn>
- PyPI: <https://pypi.org/project/pywebnn/>
- Issues: <https://github.com/rustnn/pywebnn/issues>
