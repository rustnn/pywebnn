#!/usr/bin/env python3
"""
SmolLM-135M text generation demo using a WebNN graph from Hugging Face Hub.

The demo downloads:
- model.webnn
- model.weights
- manifest.json (or model.manifest.json fallback)
- tokenizer.json

Then it runs an autoregressive decode loop with dynamic KV-cache growth.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlretrieve

import numpy as np

import webnn

try:
    from tokenizers import Tokenizer
except ImportError as exc:
    raise SystemExit(
        "This demo requires tokenizers. Install with: pip install tokenizers"
    ) from exc


DEFAULT_MODEL_ID = "tarekziade/SmolLM-135M-webnn"
DEFAULT_HF_MODEL_ID = "HuggingFaceTB/SmolLM-135M"
LAYER_KEY_RE = re.compile(r"^past_key_values_(\d+)_key$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SmolLM-135M generation demo from Hugging Face Hub WebNN files"
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hub model id")
    parser.add_argument("--prompt", default="Once upon a time", help="Prompt text")
    parser.add_argument(
        "--max-new-tokens", type=int, default=100, help="Number of tokens to generate"
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "gpu", "coreml"],
        default="cpu",
        help="Execution backend hint",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download model/tokenizer files",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print per-step argmax trace",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream decoded token pieces as they are generated",
    )
    parser.add_argument(
        "--decoding",
        choices=["greedy", "sample"],
        default="greedy",
        help="Decoding strategy",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="RNG seed used for sampling (and comparison baseline)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (used with --decoding=sample)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling (0 disables)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p / nucleus sampling in (0, 1]",
    )
    parser.add_argument(
        "--compare-transformers",
        action="store_true",
        help="Run a transformers baseline and print token-by-token comparison",
    )
    parser.add_argument(
        "--hf-model-id",
        default=DEFAULT_HF_MODEL_ID,
        help="Hugging Face model id for transformers baseline comparison",
    )
    return parser.parse_args()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted, dtype=np.float64)
    denom = np.sum(exp)
    if denom <= 0 or not np.isfinite(denom):
        return np.full_like(exp, 1.0 / float(len(exp)))
    return exp / denom


def _sample_next_token(
    logits: np.ndarray,
    rng: np.random.Generator,
    decoding: str,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    if decoding == "greedy" or temperature <= 0.0:
        return int(np.argmax(logits))

    scaled = logits.astype(np.float64) / float(temperature)

    if top_k > 0 and top_k < scaled.size:
        top_indices = np.argpartition(scaled, -top_k)[-top_k:]
        mask = np.full(scaled.shape, -np.inf, dtype=np.float64)
        mask[top_indices] = scaled[top_indices]
        scaled = mask

    probs = _softmax(scaled)

    if 0.0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        sorted_probs = probs[order]
        csum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(csum, top_p, side="right") + 1
        keep = order[:cutoff]
        filtered = np.zeros_like(probs)
        filtered[keep] = probs[keep]
        mass = np.sum(filtered)
        probs = filtered / mass if mass > 0 else probs

    return int(rng.choice(len(probs), p=probs))


def _download_file(url: str, destination: Path, force: bool) -> Path:
    if destination.exists() and not force:
        print(f"   [CACHED] {destination.name}")
        return destination
    print(f"   [DOWNLOAD] {destination.name}")
    urlretrieve(url, destination)
    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"   [OK] Downloaded {destination.name} ({size_mb:.1f} MB)")
    return destination


def resolve_model_files(model_id: str, force: bool) -> dict[str, str]:
    """Download model files via Hub, with fallback for model.manifest.json naming."""
    hub = webnn.Hub()
    try:
        files = hub.download_model(model_id, force=force)
    except RuntimeError:
        cache_dir = Path(hub.cache_dir) / model_id.replace("/", "--")
        cache_dir.mkdir(parents=True, exist_ok=True)
        base_url = f"https://huggingface.co/{model_id}/resolve/main"

        graph = _download_file(
            f"{base_url}/model.webnn?download=true",
            cache_dir / "model.webnn",
            force=force,
        )
        weights = _download_file(
            f"{base_url}/model.weights?download=true",
            cache_dir / "model.weights",
            force=force,
        )
        manifest = cache_dir / "manifest.json"
        try:
            _download_file(
                f"{base_url}/manifest.json?download=true",
                manifest,
                force=force,
            )
        except HTTPError:
            _download_file(
                f"{base_url}/model.manifest.json?download=true",
                cache_dir / "model.manifest.json",
                force=force,
            )
            manifest = cache_dir / "model.manifest.json"

        files = {
            "graph": str(graph),
            "weights": str(weights),
            "manifest": str(manifest),
        }

    return files


def resolve_tokenizer(model_id: str, cache_dir: Path, force: bool) -> Path:
    base_url = f"https://huggingface.co/{model_id}/resolve/main"
    tokenizer_path = cache_dir / "tokenizer.json"
    try:
        return _download_file(
            f"{base_url}/tokenizer.json?download=true",
            tokenizer_path,
            force=force,
        )
    except HTTPError as exc:
        raise RuntimeError(
            f"Failed to download tokenizer.json for {model_id}: {exc}"
        ) from exc


def create_context(backend: str) -> webnn.MLContext:
    ml = webnn.ML()
    if backend == "cpu":
        return ml.create_context(power_preference="default", accelerated=False)
    if backend == "gpu":
        return ml.create_context(power_preference="high-performance", accelerated=True)
    return ml.create_context(power_preference="low-power", accelerated=True)


def discover_layers(input_names: list[str]) -> list[int]:
    layers = []
    for name in input_names:
        match = LAYER_KEY_RE.match(name)
        if match:
            layers.append(int(match.group(1)))
    return sorted(set(layers))


def run_transformers_baseline(
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    decoding: str,
    seed: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[list[int], str, list[int]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers baseline requires: pip install transformers torch"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise RuntimeError("Transformers tokenizer produced empty prompt ids")

    rng = np.random.default_rng(seed)
    generated: list[int] = []

    with torch.no_grad():
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        outputs = model(input_ids=input_ids, use_cache=True)
        logits = outputs.logits[0, -1].float().cpu().numpy()
        past_key_values = outputs.past_key_values

        for _ in range(max_new_tokens):
            next_id = _sample_next_token(
                logits, rng, decoding, temperature, top_k, top_p
            )
            generated.append(next_id)

            step_ids = torch.tensor([[next_id]], dtype=torch.long)
            outputs = model(
                input_ids=step_ids, past_key_values=past_key_values, use_cache=True
            )
            logits = outputs.logits[0, -1].float().cpu().numpy()
            past_key_values = outputs.past_key_values

    return generated, tokenizer.decode(generated), prompt_ids


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("SmolLM-135M Hub Demo (WebNN)")
    print("=" * 70)
    print(f"Model:   {args.model_id}")
    print(f"Prompt:  {args.prompt}")
    print(f"Tokens:  {args.max_new_tokens}")
    print(f"Backend: {args.backend}")
    print(f"Decode:  {args.decoding}")
    if args.decoding == "sample":
        print(
            f"Sample: seed={args.seed} temperature={args.temperature} top_k={args.top_k} top_p={args.top_p}"
        )
    print()

    print("Downloading model files...")
    model_files = resolve_model_files(args.model_id, force=args.force_download)
    model_cache_dir = Path(model_files["graph"]).parent
    tokenizer_path = resolve_tokenizer(
        args.model_id, model_cache_dir, force=args.force_download
    )
    print()

    print("Loading graph...")
    graph = webnn.MLGraph.load(
        model_files["graph"],
        manifest_path=model_files["manifest"],
        weights_path=model_files["weights"],
    )
    print(f"   [OK] Operands:   {graph.operand_count}")
    print(f"   [OK] Operations: {graph.operation_count}")
    input_names = graph.get_input_names()
    output_names = graph.get_output_names()
    print(f"   [OK] Inputs:     {len(input_names)}")
    print(f"   [OK] Outputs:    {len(output_names)}")
    print()

    print("Creating context...")
    context = create_context(args.backend)
    print(f"   [OK] Context created (accelerated={context.accelerated})")
    print()

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    if not prompt_ids:
        raise RuntimeError("Prompt tokenized to an empty sequence")
    print(f"   [OK] Prompt token ids: {prompt_ids}")
    print()

    layers = discover_layers(input_names)
    if not layers:
        raise RuntimeError("No KV-cache layer inputs detected in graph")

    num_heads = 3
    head_dim = 64
    past_key_values: dict[str, np.ndarray] = {}
    for layer in layers:
        past_key_values[f"past_key_values_{layer}_key"] = np.zeros(
            (1, num_heads, 0, head_dim), dtype=np.float32
        )
        past_key_values[f"past_key_values_{layer}_value"] = np.zeros(
            (1, num_heads, 0, head_dim), dtype=np.float32
        )

    def run_step(token_id: int, position: int) -> np.ndarray:
        inputs: dict[str, np.ndarray] = {
            "input_ids": np.array([[token_id]], dtype=np.int64),
            "position_ids": np.array([[position]], dtype=np.int64),
            "attention_mask": np.ones((1, position + 1), dtype=np.int64),
        }
        inputs.update(past_key_values)

        outputs = context.compute(graph, inputs)
        logits = np.asarray(outputs["logits"], dtype=np.float32)[0, 0, :]

        for layer in layers:
            pk_name = f"present_{layer}_key"
            pv_name = f"present_{layer}_value"
            if pk_name not in outputs or pv_name not in outputs:
                raise RuntimeError(f"Missing cache outputs for layer {layer}")
            past_key_values[f"past_key_values_{layer}_key"] = np.asarray(
                outputs[pk_name], dtype=np.float32
            )
            past_key_values[f"past_key_values_{layer}_value"] = np.asarray(
                outputs[pv_name], dtype=np.float32
            )

        if args.trace:
            print(
                f"TRACE pos={position} token_in={token_id} logits_argmax={int(np.argmax(logits))}"
            )
        return logits

    print("Running generation...")
    rng = np.random.default_rng(args.seed)
    position = 0
    last_logits = None
    for token_id in prompt_ids:
        last_logits = run_step(token_id, position)
        position += 1

    if last_logits is None:
        raise RuntimeError("Failed to run prompt prefill")

    generated: list[int] = []
    if args.stream:
        print("Streaming output:")
    for _ in range(args.max_new_tokens):
        next_id = _sample_next_token(
            last_logits,
            rng,
            args.decoding,
            args.temperature,
            args.top_k,
            args.top_p,
        )
        generated.append(next_id)
        if args.stream:
            piece = tokenizer.decode([next_id])
            print(piece, end="", flush=True)
        last_logits = run_step(next_id, position)
        position += 1

    if args.stream:
        print()

    generated_text = tokenizer.decode(generated)

    print()
    print("=" * 70)
    print(f"Generated token ids ({len(generated)}):")
    print(generated)
    print("-" * 70)
    print("Generated text:")
    print(generated_text)
    print("=" * 70)

    if args.compare_transformers:
        print()
        print("=" * 70)
        print("Transformers Baseline Comparison")
        print("=" * 70)
        print(f"Baseline model: {args.hf_model_id}")
        hf_generated, hf_text, hf_prompt_ids = run_transformers_baseline(
            model_id=args.hf_model_id,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            decoding=args.decoding,
            seed=args.seed,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        print(f"Prompt ids (WebNN tokenizer):      {prompt_ids}")
        print(f"Prompt ids (transformers tokenizer): {hf_prompt_ids}")
        print("-" * 70)
        print("Token-by-token:")
        print("idx | webnn | hf | match")
        print("-" * 70)
        for i, (w, h) in enumerate(zip(generated, hf_generated)):
            status = "==" if w == h else "!="
            print(f"{i:3d} | {w:5d} | {h:5d} | {status}")
        if len(generated) != len(hf_generated):
            print(
                f"[WARNING] Different lengths: webnn={len(generated)} hf={len(hf_generated)}"
            )
        print("-" * 70)
        print("Transformers text:")
        print(hf_text)
        texts_match = generated_text == hf_text
        print("-" * 70)
        print(f"Exact text match: {'YES' if texts_match else 'NO'}")
        if not texts_match:
            print("[ERROR] WebNN and transformers generated different text output")
            print("-" * 70)
            print("Unified diff (webnn vs transformers):")
            diff = difflib.unified_diff(
                generated_text.splitlines(),
                hf_text.splitlines(),
                fromfile="webnn",
                tofile="transformers",
                lineterm="",
            )
            for line in diff:
                print(line)
            sys.exit(1)
        print("=" * 70)


if __name__ == "__main__":
    main()
