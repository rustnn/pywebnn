# Advanced usage

Keep input/output tensors and named binding dictionaries alive across inference loops.
Allocate before the loop; inside it, write changed inputs, dispatch, and read outputs.
Backend execution and validation are RustNN responsibilities.

Graphs and tensors retain their native context owner. Builders are single-use: build
consumes the recorded graph, including on finalization or backend compilation failure.
Use the native builder queries before compilation when output tensor shapes are needed.

NumPy represents int4 values using int8 carriers and uint4 values using uint8 carriers.
The binding packs/unpacks those logical values for RustNN's packed storage.

Dynamic tensor sizes use `resize_tensor` and `set_tensor_capacity`; availability depends
on RustNN's enabled features and selected backend. Python neither infers graph shapes
nor stores operation metadata.

Tensor/context ownership validation is tracked in RustNN issue #320. No Python context
guard substitutes for the pending native tensor checks.
