"""Process isolation is required for ORT's global environment."""
import os
import subprocess
import sys

import pytest


def run(code):
    env = os.environ.copy()
    env.pop("RUST_LOG", None)
    return subprocess.run(
        [sys.executable, "-c", code], check=True, env=env,
        capture_output=True, text=True,
    )


def test_custom_environment_and_duplicate():
    run("""
import webnn
webnn.init_global_ort_env([])
context = webnn.ML().create_context(backend="onnx", accelerated=False)
try:
    webnn.init_global_ort_env([])
except webnn.OrtAlreadyInitialized:
    pass
else:
    raise AssertionError("environment replaced")
""")


def test_lazy_environment_rejects_replacement():
    run("""
import webnn
context = webnn.ML().create_context(backend="onnx", accelerated=False)
try:
    webnn.init_global_ort_env([])
except webnn.OrtAlreadyInitialized:
    pass
else:
    raise AssertionError("lazy environment replaced")
""")


def test_failed_plugin_does_not_install_environment():
    run("""
from pathlib import Path
import webnn
try:
    webnn.init_global_ort_env([("missing", Path("/missing/webnn_ep.so"))], log_level="error")
except RuntimeError:
    pass
else:
    raise AssertionError("missing plugin accepted")
webnn.init_global_ort_env([], log_level="warning")
""")


@pytest.mark.parametrize("level", ["verbose", "info", "warning", "error", "fatal", "INFO", None])
def test_log_levels(level):
    run(f"""
import webnn
webnn.init_global_ort_env([], log_level={level!r})
context = webnn.ML().create_context(backend="onnx", accelerated=False)
""")


def test_invalid_log_level_does_not_initialize_environment():
    run("""
import webnn
try:
    webnn.init_global_ort_env([], log_level="invalid")
except ValueError as error:
    assert "log_level" in str(error)
else:
    raise AssertionError("invalid log level accepted")
webnn.init_global_ort_env([], log_level="warning")
""")
