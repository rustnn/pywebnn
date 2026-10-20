"""No Python-only device tensor API is exposed."""

import webnn


def test_no_device_tensor_placeholder():
    assert not hasattr(webnn._rustnn, "MLDeviceTensor")
    assert not hasattr(webnn.MLContext, "create_device_tensor")
