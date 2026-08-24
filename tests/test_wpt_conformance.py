"""
WPT WebNN Conformance Tests

Runs W3C Web Platform Tests (WPT) for WebNN against pywebnn/rustnn by loading
original `.https.any.js` conformance files through a Node.js bridge (Option A).

Requirements:
    - Node.js on PATH
    - WPT cache at ../webnnjs/.cache/wpt (or set WPT_DIR)

Usage:
    pytest tests/test_wpt_conformance.py -v
    pytest tests/test_wpt_conformance.py --wpt-backend=cpu -k "abs" -v
    pytest tests/test_wpt_conformance.py --wpt-backend=coreml -v   # macOS only
    pytest tests/test_wpt_conformance.py --wpt-backend=all -v
    WPT_BACKEND=cpu pytest tests/test_wpt_conformance.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from runtime_support import COREML_BACKEND_AVAILABLE, EXECUTION_BACKEND_AVAILABLE

try:
    from wpt_assert import assert_output_close
    from wpt_tolerance_fallback import compute_wpt_tolerance_fallback
    from wpt_execute_graph import execute_graph_resources, normalize_op_name
    from wpt_js_loader import (
        default_wpt_dir,
        discover_wpt_files,
        load_wpt_conformance_file,
        node_available,
        operation_from_wpt_file,
        resolve_wpt_tolerance,
        wpt_cache_available,
    )
except ModuleNotFoundError as err:
    # WPT support is optional: the helper modules are intentionally kept with
    # the external WPT harness and are not part of the normal test install.
    # Skip this optional test module instead of aborting collection of the
    # project test suite when that harness is unavailable.
    if err.name in {
        "wpt_assert",
        "wpt_tolerance_fallback",
        "wpt_execute_graph",
        "wpt_js_loader",
    }:
        pytest.skip(
            f"WPT conformance harness is unavailable (missing {err.name})",
            allow_module_level=True,
        )
    raise

SUPPORTED_DTYPES = {
    "float32",
    "float16",
    "int8",
    "uint8",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "int4",
    "uint4",
}


def _wpt_backends_for_config(config) -> list[str]:
    """Resolve which backend(s) to run; default is cpu only."""
    choice = (config.getoption("--wpt-backend") or os.environ.get("WPT_BACKEND") or "cpu").lower()
    if choice == "all":
        backends: list[str] = []
        if EXECUTION_BACKEND_AVAILABLE:
            backends.append("cpu")
        if COREML_BACKEND_AVAILABLE:
            backends.append("coreml")
        return backends or ["cpu"]
    return [choice]


def _backend_available(backend: str) -> bool:
    if backend == "cpu":
        return EXECUTION_BACKEND_AVAILABLE
    if backend == "coreml":
        return COREML_BACKEND_AVAILABLE
    return False


def _create_wpt_context(ml, backend_name: str):
    device_type_map = {
        "cpu": "cpu",
        "coreml": "npu",
    }
    device_type = device_type_map.get(backend_name, "auto")
    return ml.create_context(
        power_preference="default",
        accelerated=backend_name != "cpu",
        device_type=device_type,
    )


@pytest.fixture(scope="session")
def wpt_context_cache(ml):
    """One MLContext per backend for the whole WPT session (builders remain per-test)."""
    return {}


@pytest.fixture
def context(backend_name, ml, wpt_context_cache):
    """Reuse MLContext per backend; execute_graph_resources creates a fresh builder each test."""
    if not _backend_available(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not available on this platform")

    if backend_name not in wpt_context_cache:
        wpt_context_cache[backend_name] = _create_wpt_context(ml, backend_name)
    return wpt_context_cache[backend_name]


def generate_test_id(operation: str, test_case: dict[str, Any]) -> str:
    test_name = test_case.get("name", "unnamed")
    return f"{operation}::{test_name}".replace(" ", "_")


def should_skip_test(graph: dict[str, Any]) -> str | None:
    tensors = list((graph.get("inputs") or {}).values()) + list(
        (graph.get("expectedOutputs") or {}).values()
    )
    for tensor in tensors:
        descriptor = tensor.get("descriptor", tensor)
        data_type = descriptor.get("dataType")
        if data_type not in SUPPORTED_DTYPES:
            return f"unsupported dataType: {data_type}"
    return None


def pytest_generate_tests(metafunc):
    if metafunc.definition.name == "test_wpt_conformance" and "backend_name" in metafunc.fixturenames:
        backends = _wpt_backends_for_config(metafunc.config)
        metafunc.parametrize("backend_name", backends)

    if "wpt_test_case" not in metafunc.fixturenames:
        return

    if not node_available() or not wpt_cache_available():
        metafunc.parametrize(
            "wpt_test_case,wpt_file,operation",
            [(None, None, None)],
            ids=["wpt_unavailable"],
        )
        return

    wpt_dir = default_wpt_dir()
    test_params: list[Any] = []
    test_ids: list[str] = []

    for js_path in discover_wpt_files(wpt_dir):
        operation = operation_from_wpt_file(js_path)
        try:
            loaded = load_wpt_conformance_file(str(js_path), str(wpt_dir))
        except (RuntimeError, FileNotFoundError, json.JSONDecodeError) as err:
            if "No webnn_conformance_test" in str(err):
                continue
            raise RuntimeError(
                f"Failed to load WPT graph conformance file {js_path.name}: {err}"
            ) from err

        for test_case in loaded.get("tests", []):
            marks = []
            test_name = test_case.get("name", "")

            if "large" in test_name.lower():
                marks.append(pytest.mark.slow)

            graph_desc = test_case.get("graph", {})
            for input_spec in (graph_desc.get("inputs") or {}).values():
                descriptor = input_spec.get("descriptor", input_spec)
                shape = descriptor.get("shape", [])
                element_count = 1
                for dim in shape:
                    element_count *= dim
                if element_count > 100_000:
                    marks.append(pytest.mark.slow)
                    break

            if marks:
                test_params.append(
                    pytest.param(test_case, js_path, operation, marks=marks)
                )
            else:
                test_params.append((test_case, js_path, operation))
            test_ids.append(generate_test_id(operation, test_case))

    if not test_params:
        metafunc.parametrize(
            "wpt_test_case,wpt_file,operation",
            [(None, None, None)],
            ids=["no_wpt_data"],
        )
        return

    metafunc.parametrize(
        "wpt_test_case,wpt_file,operation",
        test_params,
        ids=test_ids,
    )


def test_wpt_conformance(context, backend_name, wpt_test_case, wpt_file, operation):
    if wpt_test_case is None:
        if not node_available():
            pytest.skip("Node.js is required for WPT conformance tests")
        if not wpt_cache_available():
            pytest.skip(
                f"WPT cache not found at {default_wpt_dir()}. "
                "Fetch WPT tests into webnnjs/.cache/wpt or set WPT_DIR."
            )
        pytest.skip("No WPT conformance tests loaded")

    graph = wpt_test_case.get("graph")
    if not graph:
        pytest.skip("Invalid WPT test case (missing graph)")

    test_name = wpt_test_case.get("name", "")
    skip_reason = should_skip_test(graph)
    if skip_reason:
        pytest.skip(skip_reason)

    try:
        results = execute_graph_resources(context, graph)
    except NotImplementedError as err:
        pytest.skip(str(err))
    except (ValueError, RuntimeError) as err:
        error_str = str(err)
        if "Unsupported data type" in error_str or "Unsupported feature data type" in error_str:
            pytest.skip(f"Unsupported data type: {err}")
        raise

    tolerance = wpt_test_case.get("tolerance")
    if tolerance is None:
        try:
            tolerance = resolve_wpt_tolerance(Path(wpt_file), graph, wpt_dir=default_wpt_dir())
        except RuntimeError:
            tolerance = None
    if tolerance is None:
        tolerance = compute_wpt_tolerance_fallback(graph)
    operators = graph.get("operators") or []
    graph_operator_names = [normalize_op_name(op.get("name", "")) for op in operators]
    last_op = graph_operator_names[-1] if graph_operator_names else "unknown"

    for output_name, expected_spec in (graph.get("expectedOutputs") or {}).items():
        if output_name not in results:
            pytest.fail(f"Output '{output_name}' not found in results")
        assert_output_close(
            operator_name=last_op,
            graph_operator_names=graph_operator_names,
            graph=graph,
            tolerance=tolerance,
            output_name=output_name,
            expected_spec=expected_spec,
            actual=results[output_name],
        )


pytestmark = pytest.mark.wpt
