# Python API

The public classes are native wrappers: `ML`, `MLContext`, `MLGraphBuilder`,
`MLOperand`, `MLGraph`, and `MLTensor`. The type stubs in `python/webnn/_rustnn.pyi`
list the full operation signatures; the public stub re-exports those definitions.

## Context and tensors

- `ML.create_context(power_preference="default", accelerated=True, backend="auto")`
  adapts arguments to RustNN's context options and backend hint.
- `context.create_graph_builder()` creates a runtime-bound native builder.
- `context.create_tensor(shape, data_type, readable=False, writable=False)`
  explicitly allocates a native tensor.
- `context.write_tensor(tensor, data)` and `context.read_tensor(tensor)` adapt NumPy
  buffers to native writes and reads without changing the native tensor dtype.
- `context.dispatch(graph, inputs, outputs)` forwards named tensor bindings.
- `context.resize_tensor(tensor, shape)` and `set_tensor_capacity(tensor, max_shape)`
  forward RustNN's tensor extensions.
- `context.accelerated` queries the native context.

RustNN has not implemented capability reporting or tensor destruction; corresponding
Python calls raise `NotImplementedError` rather than simulating those operations.

## Builder

Operations forward to their RustNN counterparts. Numeric coefficients retain double
precision; `arg_min` and `arg_max` default to int32; quantization zero points are
optional. Each operation accepts a keyword-only `label`.

`builder.input(name, shape, data_type)` creates an input handle. `builder.constant`
adapts NumPy values to native constant bytes, converting an explicit dtype override.
Reshape/expand shapes and slice sizes accept integer dimensions or named dynamic
dimensions such as `{"name": "batch", "maxSize": 8}`. RustNN determines their semantics.

Use `builder.operand_shape(operand)` and `builder.operand_data_type(operand)` to query
native metadata **before** build. Python operands are opaque handles with no cached
shape, dtype, or name properties.

`builder.save(outputs, path)` forwards RustNN's native save method before compilation.
`builder.build(outputs)` calls RustNN's existing build method and consumes the builder.
Subsequent native builder queries fail as they do in RustNN.

## Compiled graph

`MLGraph` is a native compiled handle used by `context.dispatch`. It retains its
runtime owner for lifetime safety. Python stores no `GraphInfo` or separate graph
representation. Its `input_descriptors` and `output_descriptors` properties forward
RustNN's public I/O fields as dictionaries with `shape` and `data_type` entries.
Dynamic dimensions remain `{ "name": ..., "maxSize": ... }` dictionaries.

`context.load_graph(path)` forwards RustNN's file loader and compiles the loaded
graph for that context. RustNN resolves external weights relative to the graph file.
Allocate tensors explicitly from the native I/O descriptors and reuse them across
dispatches. Download Hub files with the independent `webnn.Hub` helper or
`huggingface_hub`. `Hub` provides `download_model`, `list_cached_models`, and
`clear_cache`; it performs no graph parsing or validation.
