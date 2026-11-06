#!/usr/bin/env python3
"""
Resolve the ONNX Runtime dynamic library path for ORT_DYLIB_PATH.
Prints the selected absolute path to stdout.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from pathlib import Path


def _is_core_ort_library(path: str) -> bool:
    name = os.path.basename(path).lower()
    if any(token in name for token in ("providers", "pybind", "extensions")):
        return False
    if sys.platform == "darwin":
        return name.startswith("libonnxruntime") and name.endswith(".dylib")
    if sys.platform == "win32":
        return name.startswith("onnxruntime") and name.endswith(".dll")
    return name.startswith("libonnxruntime.so")


def _exports_ort_api_base(path: str) -> bool:
    try:
        lib = ctypes.CDLL(path)
        return hasattr(lib, "OrtGetApiBase")
    except Exception:
        return False


def _patterns() -> list[str]:
    if sys.platform == "darwin":
        return ["libonnxruntime*.dylib"]
    if sys.platform == "win32":
        return ["onnxruntime*.dll"]
    return ["libonnxruntime.so*", "libonnxruntime*.so*"]


def main() -> int:
    try:
        import onnxruntime  # type: ignore
    except ImportError:
        print("onnxruntime package is not installed", file=sys.stderr)
        return 1

    package_dir = Path(onnxruntime.__file__).resolve().parent
    search_dirs = [package_dir / "capi", package_dir]

    candidates: list[str] = []
    for base in search_dirs:
        if not base.exists():
            continue
        for pattern in _patterns():
            candidates.extend(glob.glob(str(base / pattern)))

    core = sorted(
        {str(Path(p).resolve()) for p in candidates if _is_core_ort_library(p)},
        key=lambda p: (len(os.path.basename(p)), os.path.basename(p)),
    )
    for candidate in core:
        if _exports_ort_api_base(candidate):
            print(candidate)
            return 0

    print("Could not resolve a valid ONNX Runtime dylib (OrtGetApiBase missing)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
