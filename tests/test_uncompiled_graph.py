"""Export without a runtime, then load or execute the same native graph."""

import numpy as np
import pytest
import webnn
from runtime_support import requires_execution_backend


def make_graph():
    builder = webnn.MLGraphBuilder.new_uncompiled()
    x = builder.input("x", [2, 3], "float32")
    bias = builder.constant(np.array([1, 2, 3], dtype=np.float32))
    output = builder.add(x, bias)
    outputs = {"y": output}
    return builder, x, bias, output, outputs


def test_export_without_runtime(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Export must not initialize a runtime")

    monkeypatch.setattr(webnn.ML, "create_context", forbidden)
    builder, x, bias, output, outputs = make_graph()
    assert x.shape == [2, 3]
    assert bias.shape == [3]
    assert output.shape == [2, 3]
    assert output.data_type == "float32"
    assert "webnn_graph" in builder.rustnn_webnn_text_for_outputs(outputs)
    path = tmp_path / "model.webnn"
    builder.rustnn_save_webnn(outputs, path)
    assert "@weights(" in path.read_text()
    assert path.with_suffix(".safetensors").is_file()
    loaded = webnn.MLGraph.load(path)
    graph = builder.build(outputs)
    assert graph.input_descriptors == {"x": {"shape": [2, 3], "data_type": "float32"}}
    assert graph.output_descriptors == {"y": {"shape": [2, 3], "data_type": "float32"}}
    assert loaded.input_descriptors == graph.input_descriptors
    assert loaded.output_descriptors == graph.output_descriptors
    # Metadata remains usable even after its builder has finalized.
    shape = output.shape
    shape[0] = 99
    assert output.shape == [2, 3]
    with pytest.raises(AttributeError):
        output.data_type = "int32"
    with pytest.raises(RuntimeError, match="already called"):
        builder.build(outputs)
    with pytest.raises(RuntimeError, match="already called"):
        builder.rustnn_webnn_text_for_outputs(outputs)
    with pytest.raises(RuntimeError, match="already called"):
        builder.rustnn_save_webnn(outputs, path)


@requires_execution_backend
def test_lazy_execution_and_serialization(tmp_path):
    builder, _, _, _, outputs = make_graph()
    path = tmp_path / "model.webnn"
    builder.rustnn_save_webnn(outputs, path)
    graph = builder.build(outputs)
    context = webnn.ML().create_context(accelerated=False)
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    expected = data + np.array([1, 2, 3], dtype=np.float32)
    source = context.create_tensor([2, 3], "float32", writable=True)
    target = context.create_tensor([2, 3], "float32", readable=True)
    for value in (data, data * 2):
        context.write_tensor(source, value)
        context.dispatch(graph, {"x": source}, {"y": target})
        actual = context.read_tensor(target)
        np.testing.assert_array_equal(actual, value + [1, 2, 3])
        assert actual.dtype == np.float32
    loaded = webnn.MLGraph.load(path)
    context.write_tensor(source, data)
    context.dispatch(loaded, {"x": source}, {"y": target})
    np.testing.assert_array_equal(context.read_tensor(target), expected)
    other_context = webnn.ML().create_context(accelerated=False)
    with pytest.raises(ValueError, match="different MLContext"):
        other_context.dispatch(graph, {"x": source}, {"y": target})


@requires_execution_backend
def test_dispatch_scalar_output():
    builder = webnn.MLGraphBuilder.new_uncompiled()
    x = builder.input("x", [3], "float32")
    graph = builder.build({"y": builder.reduce_sum(x, keep_dimensions=False)})
    context = webnn.ML().create_context(accelerated=False)
    source = context.create_tensor([3], "float32", writable=True)
    target = context.create_tensor([], "float32", readable=True)
    context.write_tensor(source, np.array([1, 2, 3], dtype=np.float32))
    context.dispatch(graph, {"x": source}, {"y": target})
    output = context.read_tensor(target)
    assert output.shape == ()
    assert output == 6
