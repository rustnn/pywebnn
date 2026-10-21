#!/usr/bin/env python3
"""
T5-small encoder/decoder WebNN demo.

Loads two WebNN graphs (encoder and decoder) from a local directory and runs a
greedy decode loop to generate text. The expected directory layout is:

    encoder_model.webnn
    encoder_model.safetensors (or manifest-backed external weights)
    decoder_model.webnn
    decoder_model.safetensors (or manifest-backed external weights)

By default, the script points to models/t5-small-webnn. Override
with the T5_WEBNN_DIR environment variable or --model-dir flag.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import webnn
from model_files import max_shape

MAX_SEQ_LEN = 128
DECODER_SEQ_LEN = 1
NUM_LAYERS = 6
NUM_HEADS = 8
HEAD_DIM = 64


DEFAULT_MODEL_DIR = Path(os.environ.get("T5_WEBNN_DIR", "models/t5-small-webnn"))


def _load_graph(context: webnn.MLContext, graph_path: Path) -> webnn.MLGraph:
    """Delegate graph parsing, weights, and compilation to RustNN."""
    return context.load_graph(str(graph_path))


def _create_context(device: str) -> webnn.MLContext:
    """Create a WebNN context with a simple CPU/GPU toggle."""
    ml = webnn.ML()
    if device == "cpu":
        return ml.create_context(power_preference="default", accelerated=False)
    return ml.create_context(power_preference="high-performance", accelerated=True)


def _select_output(outputs: Dict[str, np.ndarray], desired: str) -> np.ndarray:
    """Return the named output, falling back to the first tensor if missing."""
    if desired in outputs:
        return outputs[desired]
    if outputs:
        return outputs[next(iter(outputs))]
    raise RuntimeError("No outputs returned from graph execution")


def run_encoder(
    context: webnn.MLContext,
    graph: webnn.MLGraph,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    """Execute the encoder and return hidden states."""
    if input_ids.dtype != np.int64:
        input_ids = input_ids.astype(np.int64, copy=False)
    if attention_mask.dtype != np.int64:
        attention_mask = attention_mask.astype(np.int64, copy=False)
    sources = {
        name: context.create_tensor(max_shape(spec), spec["data_type"], writable=True)
        for name, spec in graph.input_descriptors.items()
    }
    targets = {
        name: context.create_tensor(max_shape(spec), spec["data_type"], readable=True)
        for name, spec in graph.output_descriptors.items()
    }
    data = {"input_ids": input_ids, "attention_mask": attention_mask}
    for name, tensor in sources.items():
        context.write_tensor(tensor, data[name])
    context.dispatch(graph, sources, targets)
    result = {name: context.read_tensor(tensor) for name, tensor in targets.items()}
    return _select_output(result, "last_hidden_state")


def run_decoder(
    context: webnn.MLContext,
    graph: webnn.MLGraph,
    graph_inputs: list[str],
    decoder_input_ids: np.ndarray,
    encoder_hidden_states: np.ndarray,
    encoder_attention_mask: np.ndarray,
    input_tensors: dict[str, webnn.MLTensor],
    output_tensors: dict[str, webnn.MLTensor],
    past_inputs: Dict[str, np.ndarray] | None = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Execute the decoder and return logits and present key/values."""
    if decoder_input_ids.dtype != np.int64:
        decoder_input_ids = decoder_input_ids.astype(np.int64, copy=False)
    if encoder_attention_mask.dtype != np.int64:
        encoder_attention_mask = encoder_attention_mask.astype(np.int64, copy=False)
    # Initialize empty KV-cache inputs if the graph expects them.
    inputs: Dict[str, np.ndarray] = {
        "input_ids": decoder_input_ids,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
    }
    if past_inputs:
        for name, value in past_inputs.items():
            if name in graph_inputs:
                inputs[name] = value

    dimensions = {}
    input_specs, output_specs = graph.input_descriptors, graph.output_descriptors
    for name, tensor in input_tensors.items():
        value = inputs[name]
        spec = input_specs[name]
        for dimension, size in zip(spec["shape"], value.shape):
            if isinstance(dimension, dict):
                dimensions[dimension["name"]] = size
        if list(value.shape) != tensor.shape:
            context.resize_tensor(tensor, list(value.shape))
        context.write_tensor(tensor, value)
    for name, tensor in output_tensors.items():
        shape = [
            (
                dimensions.get(dim["name"], dim["maxSize"])
                if isinstance(dim, dict)
                else dim
            )
            for dim in output_specs[name]["shape"]
        ]
        if shape != tensor.shape:
            context.resize_tensor(tensor, shape)
    context.dispatch(graph, input_tensors, output_tensors)
    result = {
        name: context.read_tensor(tensor) for name, tensor in output_tensors.items()
    }
    logits = _select_output(result, "logits")
    presents = {
        name: tensor for name, tensor in result.items() if name.startswith("present_")
    }
    return logits, presents


def greedy_decode(
    context: webnn.MLContext,
    encoder_graph: webnn.MLGraph,
    decoder_graph: webnn.MLGraph,
    tokenizer: AutoTokenizer,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    max_new_tokens: int,
    decoder_inputs: list[str],
    debug: bool = False,
) -> str:
    """Greedy decoding with decoder KV-cache."""
    encoder_hidden = run_encoder(context, encoder_graph, input_ids, attention_mask)
    has_cache = any("past_key_values" in name for name in decoder_inputs)

    # Decoder starts with the pad token for T5 (acts as BOS).
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    start_id = pad_id  # T5 uses pad token as decoder start
    eos_id = tokenizer.eos_token_id or pad_id

    # Initialize past caches (decoder + encoder).
    past_inputs: Dict[str, np.ndarray] = {}
    if has_cache:
        for layer in range(NUM_LAYERS):
            past_inputs[f"past_key_values_{layer}_decoder_key"] = np.zeros(
                (1, NUM_HEADS, DECODER_SEQ_LEN, HEAD_DIM), dtype=np.float32
            )
            past_inputs[f"past_key_values_{layer}_decoder_value"] = np.zeros(
                (1, NUM_HEADS, DECODER_SEQ_LEN, HEAD_DIM), dtype=np.float32
            )
            past_inputs[f"past_key_values_{layer}_encoder_key"] = np.zeros(
                (1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM), dtype=np.float32
            )
            past_inputs[f"past_key_values_{layer}_encoder_value"] = np.zeros(
                (1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM), dtype=np.float32
            )

    # Allocate maximum storage once, then resize bindings within native capacity.
    input_tensors = {
        name: context.create_tensor(max_shape(spec), spec["data_type"], writable=True)
        for name, spec in decoder_graph.input_descriptors.items()
    }
    output_tensors = {
        name: context.create_tensor(max_shape(spec), spec["data_type"], readable=True)
        for name, spec in decoder_graph.output_descriptors.items()
    }
    generated = [start_id]
    for _ in range(max_new_tokens):
        decoder_input = np.full((1, DECODER_SEQ_LEN), pad_id, dtype=np.int64)
        decoder_input[0, -1] = generated[-1]
        logits, presents = run_decoder(
            context,
            decoder_graph,
            decoder_inputs,
            decoder_input_ids=decoder_input,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=attention_mask,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            past_inputs=past_inputs,
        )
        # Use the last position for next-token selection.
        last_logits = logits[0, min(len(generated) - 1, logits.shape[1] - 1)]
        if debug:
            topk = np.argsort(last_logits)[-5:][::-1]
            decoded_topk = [tokenizer.decode([int(t)]) for t in topk]
            print(
                "[DEBUG] step={step} top5 tokens: ".format(step=len(generated))
                + ", ".join(f"{int(t)}='{s}'" for t, s in zip(topk, decoded_topk))
            )
        next_token = int(np.argmax(last_logits))
        generated.append(next_token)
        if eos_id is not None and next_token == eos_id:
            break

        # Feed present -> past for next iteration (decoder cache). Encoder past
        # stays static because encoder_hidden_states are provided each step.
        if has_cache:
            for layer in range(NUM_LAYERS):
                for kind in ("key", "value"):
                    present_name = f"present_{layer}_decoder_{kind}"
                    past_name = f"past_key_values_{layer}_decoder_{kind}"
                    if present_name in presents:
                        past_inputs[past_name] = presents[present_name]

    return tokenizer.decode(generated, skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T5-small WebNN encoder/decoder demo")
    parser.add_argument(
        "--prompt",
        default="translate English to German: How are you doing today?",
        help="Input prompt for T5",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing encoder/decoder WebNN files",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Preferred device hint for MLContext",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Skip running the transformers reference model",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose decoding debug output (top-5 tokens each step)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cache-enabled decoder_past_model.* if present",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # Tokenize input text (shared by WebNN and reference).
    encoded_np = tokenizer(
        [args.prompt],
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="np",
    )
    input_ids = encoded_np["input_ids"].astype(np.int64)
    attention_mask = encoded_np["attention_mask"].astype(np.int64)

    # Run transformers reference for comparison (CPU).
    if not args.skip_reference:
        print("[INFO] Running transformers reference (PyTorch, CPU)...")
        encoded_torch = tokenizer(
            [args.prompt],
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        ref_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        ref_output = ref_model.generate(
            **encoded_torch, max_new_tokens=args.max_new_tokens
        )
        ref_text = tokenizer.batch_decode(ref_output, skip_special_tokens=True)[0]
        print("[OK] Transformers output:")
        print(ref_text.strip())
    else:
        ref_text = None

    context = _create_context(args.device)
    # RustNN resolves each graph's external weights beside its graph file.
    # Load WebNN graphs.
    encoder_graph = _load_graph(
        context,
        args.model_dir / "encoder_model.webnn",
    )
    # Prefer cache-enabled decoder export if present.
    decoder_webnn = args.model_dir / "decoder_model.webnn"
    if args.use_cache:
        candidate_webnn = args.model_dir / "decoder_past_model.webnn"
        if candidate_webnn.exists():
            decoder_webnn = candidate_webnn
        else:
            print(
                "[WARN] Cache-enabled decoder_past_model.* not found; using decoder_model.*"
            )

    decoder_graph = _load_graph(context, decoder_webnn)
    decoder_inputs = list(decoder_graph.input_descriptors)

    print(f"[INFO] WebNN context created (accelerated={context.accelerated})")
    print(f"[INFO] Encoder inputs: {list(encoder_graph.input_descriptors)}")
    print(f"[INFO] Decoder inputs: {decoder_inputs}")

    output_text = greedy_decode(
        context,
        encoder_graph,
        decoder_graph,
        tokenizer,
        input_ids,
        attention_mask,
        args.max_new_tokens,
        decoder_inputs,
        debug=args.debug,
    )

    print("\n=== T5-small WebNN output ===")
    print(output_text.strip())
    if ref_text is not None:
        print("\n=== Reference vs WebNN ===")
        print(f"Reference: {ref_text.strip()}")
        print(f"WebNN    : {output_text.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
