#!/usr/bin/env python3
"""Run the Python test suite using the active virtual-environment interpreter."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    resolver = root / "tools" / "resolve_ort_dylib.py"
    resolved = subprocess.run(
        [sys.executable, str(resolver)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    ort_dylib_path = resolved.stdout.strip()
    if not ort_dylib_path:
        raise RuntimeError("ONNX Runtime library resolver returned an empty path")

    print(f"Using ORT_DYLIB_PATH={ort_dylib_path}", flush=True)
    environment = os.environ.copy()
    environment["ORT_DYLIB_PATH"] = ort_dylib_path
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        cwd=root,
        env=environment,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
