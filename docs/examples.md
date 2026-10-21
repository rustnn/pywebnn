# Examples

Run after installing the native extension:

```sh
python examples/python_simple.py
python examples/python_matmul.py
python examples/dispatch_example.py
python examples/save_load_example.py
python examples/test_quantization_roundtrip.py
python examples/compare_quantization_sizes.py
python examples/kv_cache_device_tensors.py
python examples/text_generation_gpt.py --tokens 3 --temperature 0
python examples/text_generation_enhanced.py --use-kv-cache --tokens 3 --temperature 0
python examples/image_classification.py examples/images/test.jpg --backend cpu
```

These examples use native graphs and explicit tensor allocation. Decode loops reuse
compiled graphs and tensors. `save_load_example.py` and
`test_quantization_roundtrip.py` reload through RustNN's loader. The size comparison
includes text and external weight files.

The historical `kv_cache_device_tensors.py` filename is retained. It compares host
round trips with native ping-pong tensor bindings; RustNN determines storage and
transfers. It does not claim zero-copy device access. The enhanced generation demo
still uses its original simplified cache simulation, rather than full attention.

## Pretrained models

Install example dependencies with:

```sh
pip install pillow huggingface_hub transformers torch tokenizers
```

| Example | Model source |
| --- | --- |
| `mobilenetv2_real.py` | Local first-convolution and classifier `.npy` weights; intermediate weights are random |
| `mobilenetv2_complete.py` | Full local MobileNetV2 `.npy` weights in `examples/mobilenetv2_weights` |
| `mobilenetv2_from_hub.py` | MobileNetV2 graph and weights from Hugging Face Hub |
| `minilm_embeddings.py` | Local or Hub MiniLM graph and weights, compared with Transformers |
| `compare_minilm_onnx.py` | Local ONNX and optional WebNN model directories |
| `smollm_from_hub.py` | Hub SmolLM graph, weights, and tokenizer with dynamic KV-cache bindings |
| `t5_small_webnn.py` | Local encoder/decoder graphs selected with `--model-dir` |

Hub downloads use `huggingface_hub` in example code. File-based examples call
`context.load_graph(path)`; RustNN parses the graph, resolves external weights, and
compiles it. Keep the graph's referenced weights alongside the graph, in the format
RustNN supports. Native loader/backend errors propagate unchanged. For example:

```sh
python examples/mobilenetv2_from_hub.py examples/images/test.jpg --backend cpu
python examples/minilm_embeddings.py
python examples/smollm_from_hub.py --backend cpu --max-new-tokens 15
python examples/t5_small_webnn.py --model-dir /path/to/t5-model --skip-reference
```

Individual Hub targets are available as `make mobilenet-demo-hub`,
`make minilm-demo-hub`, and `make smollm-demo-hub`. `make run-all-demos` runs local
builder examples without downloading pretrained models.

## Training demonstrations

`train_simple_demo.py` trains a toy character model in NumPy.
`train_text_model.py` retains its NumPy training workflow and saves weights for
`text_generation_gpt.py`. Training does not run through WebNN.
