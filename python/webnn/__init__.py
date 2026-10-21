"""
WebNN Python API

This package provides Python bindings for the WebNN (Web Neural Network) API,
allowing you to build, validate, and execute neural network graphs.

Example usage:
    >>> import webnn
    >>> ml = webnn.ML()
    >>> context = ml.create_context(accelerated=False)
    >>> builder = context.create_graph_builder()
    >>>
    >>> # Build a simple graph
    >>> x = builder.input("x", [2, 3], "float32")
    >>> y = builder.input("y", [2, 3], "float32")
    >>> z = builder.add(x, y)
    >>> output = builder.relu(z)
    >>>
    >>> # Compile the graph
    >>> graph = builder.build({"output": output})
"""

import ctypes
import os
import sys

# Configure the native runtime library path when needed (implementation detail).
# Users may set WEBNN_RUNTIME_LIBRARY before importing webnn.
_RUNTIME_LIBRARY_ENV = "WEBNN_RUNTIME_LIBRARY"
if _RUNTIME_LIBRARY_ENV in os.environ:
    os.environ.setdefault("ORT_DYLIB_PATH", os.environ[_RUNTIME_LIBRARY_ENV])

if "ORT_DYLIB_PATH" not in os.environ:

    def _is_core_runtime_library(path: str) -> bool:
        name = os.path.basename(path).lower()
        if any(token in name for token in ("providers", "pybind", "extensions")):
            return False
        if sys.platform == "darwin":
            return name.startswith("libonnxruntime") and name.endswith(".dylib")
        if sys.platform == "win32":
            return name.startswith("onnxruntime") and name.endswith(".dll")
        return name.startswith("libonnxruntime.so")

    def _exports_runtime_api(path: str) -> bool:
        try:
            lib = ctypes.CDLL(path)
            return hasattr(lib, "OrtGetApiBase")
        except Exception:
            return False

    def _pick_runtime_library(candidates):
        core_candidates = [p for p in candidates if _is_core_runtime_library(p)]
        core_candidates.sort(
            key=lambda p: (len(os.path.basename(p)), os.path.basename(p))
        )
        for candidate in core_candidates:
            if _exports_runtime_api(candidate):
                return candidate
        return None

    try:
        import glob

        import onnxruntime

        runtime_file = getattr(onnxruntime, "__file__", None)
        runtime_package_dir = (
            os.path.dirname(os.path.abspath(str(runtime_file))) if runtime_file else ""
        )

        if runtime_package_dir:
            capi_dir = os.path.join(runtime_package_dir, "capi")

            if sys.platform == "darwin":
                lib_pattern = "libonnxruntime*.dylib"
            elif sys.platform == "win32":
                lib_pattern = "onnxruntime*.dll"
            else:
                lib_pattern = "libonnxruntime*.so*"

            if os.path.exists(capi_dir):
                libs = glob.glob(os.path.join(capi_dir, lib_pattern))
                selected = _pick_runtime_library(libs)
                if selected:
                    os.environ["ORT_DYLIB_PATH"] = selected

            if "ORT_DYLIB_PATH" not in os.environ:
                libs = glob.glob(os.path.join(runtime_package_dir, lib_pattern))
                selected = _pick_runtime_library(libs)
                if selected:
                    os.environ["ORT_DYLIB_PATH"] = selected

    except ImportError:
        pass

from ._rustnn import (
    ML,
    MLContext,
    MLGraphBuilder,
    MLOperand,
    MLGraph,
    MLTensor,
)
from .hub import Hub

__all__ = [
    "ML",
    "MLContext",
    "MLGraphBuilder",
    "MLOperand",
    "MLGraph",
    "MLTensor",
    "Hub",
]
__version__ = "0.5.13"
