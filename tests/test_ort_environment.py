"""Process isolation is required for ORT's global environment."""
import os
import subprocess
import sys


def run(code):
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())


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
    webnn.init_global_ort_env([("missing", Path("/missing/webnn_ep.so"))])
except RuntimeError:
    pass
else:
    raise AssertionError("missing plugin accepted")
webnn.init_global_ort_env([])
""")
