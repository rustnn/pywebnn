"""
WPT (Web Platform Tests) Utilities for WebNN

Helpers for converting WPT tensor specs to NumPy arrays and formatting failures.
Test data is loaded from WPT JavaScript via tests/wpt_js_loader.py (Node bridge).

Based on: https://github.com/web-platform-tests/wpt/tree/master/webnn
"""

import math
from typing import Any, Dict, List

import numpy as np


def format_test_failure(
    test_name: str,
    failures: List[Dict[str, Any]],
    max_failures_shown: int = 5,
) -> str:
    """Format test failure details for human-readable output."""
    lines = [f"\n❌ Test failed: {test_name}"]
    lines.append(f"   Total failures: {len(failures)}")
    lines.append(f"   Showing first {min(len(failures), max_failures_shown)} failures:")

    for i, failure in enumerate(failures[:max_failures_shown]):
        if "ulp_distance" in failure:
            tol = failure.get("tolerance", failure.get("ulp_tolerance", "?"))
            lines.append(
                f"   [{i}] index={failure['index']}: "
                f"actual={failure['actual']:.6f}, expected={failure['expected']:.6f}, "
                f"ULP={failure['ulp_distance']} (tolerance={tol})"
            )
        elif "absolute_difference" in failure:
            tol = failure.get("tolerance", failure.get("abs_tolerance", "?"))
            lines.append(
                f"   [{i}] index={failure['index']}: "
                f"actual={failure['actual']:.6f}, expected={failure['expected']:.6f}, "
                f"diff={failure['absolute_difference']:.2e} (tolerance={tol})"
            )
        else:
            lines.append(f"   [{i}] {failure}")

    if len(failures) > max_failures_shown:
        lines.append(f"   ... and {len(failures) - max_failures_shown} more failures")

    return "\n".join(lines)


def _clamp_wpt_int64(value: int) -> int:
    """Clamp out-of-range WPT/JS Number literals to int64 (e.g. INT64_MIN approximations)."""
    info = np.iinfo(np.int64)
    if value > info.max:
        return int(info.max)
    if value < info.min:
        return int(info.min)
    return value


def _parse_wpt_int64_scalar(value: Any) -> int:
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("n"):
            return _clamp_wpt_int64(int(text[:-1]))
        return _clamp_wpt_int64(int(text))
    if isinstance(value, int):
        return _clamp_wpt_int64(value)
    if isinstance(value, float):
        return _clamp_wpt_int64(int(value))
    raise TypeError(f"cannot parse int64 WPT value: {value!r}")


def convert_bigint_values(data: Any) -> Any:
    """Recursively convert JavaScript bigint literals (strings ending with 'n') to int."""
    if isinstance(data, str) and data.endswith("n"):
        try:
            return int(data[:-1])
        except ValueError:
            return data
    if isinstance(data, list):
        return [convert_bigint_values(item) for item in data]
    return data


def numpy_array_from_test_data(test_data: Dict[str, Any]) -> np.ndarray:
    """Create a NumPy array from a WPT tensor spec (descriptor + data)."""
    data = convert_bigint_values(test_data["data"])

    if "descriptor" in test_data:
        descriptor = test_data["descriptor"]
        shape = descriptor["shape"]
        dtype_str = descriptor.get("dataType", "float32")
    else:
        shape = test_data["shape"]
        dtype_str = test_data.get("dataType", "float32")

    dtype_map = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "uint32": np.uint32,
        "int8": np.int8,
        "uint8": np.uint8,
        "int64": np.int64,
        "uint64": np.uint64,
    }
    np_dtype = dtype_map.get(dtype_str, np.float32)

    if isinstance(data, (int, float)):
        total_elements = math.prod(shape) if shape else 1
        fill = _parse_wpt_int64_scalar(data) if dtype_str == "int64" else data
        if dtype_str == "uint64":
            fill = int(data)
        return np.full(shape, fill, dtype=np_dtype)

    if dtype_str == "int64":
        return np.array([_parse_wpt_int64_scalar(v) for v in data], dtype=np_dtype).reshape(shape)

    return np.array(data, dtype=np_dtype).reshape(shape)
