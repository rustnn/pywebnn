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
native metadata **before** build. Read-only `operand.shape` and `operand.data_type`
retain RustNN's inferred metadata and remain available after build.

`MLGraphBuilder.new_uncompiled()` records without creating a runtime context.
`builder.build(outputs)` finalizes a native graph and consumes the builder for
further operations. A context builder compiles immediately; an uncompiled builder
defers compilation until the graph's first execution.

`builder.save(outputs, path)` and its alias `rustnn_save_webnn(outputs, path)` write
text plus sibling SafeTensors weights. `rustnn_webnn_text_for_outputs(outputs)`
returns debug text with inline constants. Serialize before calling `build()`, which consumes the native builder.

## Graph

`MLGraph` holds a native `GraphInfo` until compilation and retains its runtime
owner after compilation for lifetime safety. Its `input_descriptors` and
`output_descriptors` properties expose native I/O fields as dictionaries with
`shape` and `data_type` entries, including before compilation.
Dynamic dimensions remain `{ "name": ..., "maxSize": ... }` dictionaries.

`context.load_graph(path)` forwards RustNN's file loader and compiles the loaded
graph for that context. RustNN resolves external weights relative to the graph file.
`MLGraph.load(path)` loads without a runtime and compiles at first dispatch. Once compiled, a graph belongs to that executing context.
Allocate tensors explicitly from the native I/O descriptors and reuse them across
dispatches. Download Hub files with the independent `webnn.Hub` helper or
`huggingface_hub`. `Hub` provides `download_model`, `list_cached_models`, and
`clear_cache`; it performs no graph parsing or validation.

## ONNX Runtime initialization

`init_global_ort_env(execution_provider_libraries, *, log_level=None)` installs
an ORT environment before device discovery or context creation. Libraries are
`(registration_name, library_path)` pairs. The optional severity threshold accepts
`"verbose"`, `"info"`, `"warning"`, `"error"`, and `"fatal"` (case-insensitive),
and applies before plugin registration. `None` retains ORT's logging defaults.
Invalid levels raise `ValueError` without initializing the runtime. Library errors
raise `RuntimeError`; replacing an installed environment raises
`OrtAlreadyInitialized`.
