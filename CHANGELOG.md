# Changelog

All notable changes to this project will be documented in this file.

## [0.5.12] - 2026-05-04

- Switch `rustnn` dependency to crates.io `"0.5.12"` (was a git tag).
- Pin `webnn-graph` dependency to crates.io `"0.3"` (was a git branch).
- Update to new ONNX split protobuf + weight API and `MLOperandDataType` enum (#14).
- Remove the use of `shape_inference` `OperatorOptions` (#13).

## [0.5.11]

- Add support for SafeTensors (#5).
- Update `pyo3` to the newest version (#9).
- Fix build after refactor (#11).
- Add `MLDimension` to `GraphBuilder` `MLOperation` (#7).
- Update PyWebNN to new RustNN `OperatorOptions` API (#6).
- Add flexible input support (#4).
