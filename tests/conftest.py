"""
Shared pytest fixtures for all test files.

This module provides common fixtures for WebNN API testing.
"""

import pytest

from runtime_support import (
    COREML_BACKEND_AVAILABLE,
    EXECUTION_BACKEND_AVAILABLE,
    requires_coreml_backend,
    requires_execution_backend,
)

# Re-export markers for tests that import from conftest.
__all__ = [
    "requires_execution_backend",
    "requires_coreml_backend",
    "EXECUTION_BACKEND_AVAILABLE",
    "COREML_BACKEND_AVAILABLE",
]


# Pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "wpt: WebNN W3C Web Platform Tests")
    config.addinivalue_line(
        "markers",
        "requires_execution_backend: Test requires a working MLContext execution backend",
    )
    config.addinivalue_line(
        "markers",
        "requires_coreml_backend: Test requires CoreML (NPU) backend on macOS",
    )
    config.addinivalue_line("markers", "slow: Tests with large inputs that take longer to run")


def pytest_addoption(parser):
    parser.addoption(
        "--wpt-backend",
        action="store",
        default=None,
        choices=["cpu", "coreml", "all"],
        help="Backend for WPT conformance tests (default: cpu, or WPT_BACKEND env)",
    )


@pytest.fixture(scope="session")
def ml():
    """Create ML instance (session-scoped)."""
    try:
        import webnn
    except ImportError:
        pytest.skip("webnn not built yet")
    return webnn.ML()


@pytest.fixture(
    params=[
        pytest.param("cpu", id="cpu")
        if EXECUTION_BACKEND_AVAILABLE
        else pytest.param(None, id="no_cpu_backend", marks=pytest.mark.skip),
        pytest.param("coreml", id="coreml")
        if COREML_BACKEND_AVAILABLE
        else pytest.param(None, id="no_coreml_backend", marks=pytest.mark.skip),
    ]
)
def backend_name(request):
    """Return the backend name for the current test."""
    return request.param


@pytest.fixture
def context(backend_name, ml):
    """Create ML context for the specified backend.

    Uses WebNN device_type hints (cpu / npu), not implementation-specific names.
    """
    if backend_name is None:
        pytest.skip("Backend not available")

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


@pytest.fixture
def builder(context):
    """Create graph builder."""
    return context.create_graph_builder()
