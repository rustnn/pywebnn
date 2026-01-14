"""
Quick check to see where embeddings diverge for all-MiniLM-L12-v2.

Runs three baselines:
1) Reference embeddings from Hugging Face transformers (PyTorch)
2) Embeddings from the original ONNX export (model-static.onnx)
3) Optional: embeddings from another ONNX file (e.g., WebNN -> ONNX reconversion)

Use the outputs to decide whether the mismatch comes from:
- The original ONNX export
- The ONNX -> WebNN conversion
- The WebNN -> ONNX round-trip or execution
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModel, AutoTokenizer
import webnn

HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Default to the local path mentioned in the instructions if it exists.
DEFAULT_STATIC_ONNX = Path("/Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn/model-static.onnx")
DEFAULT_WEBNN_DIR = Path("/Users/tarekziade/Dev/all-MiniLM-L12-v2-webnn")


def mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean pooling with attention mask (matches sentence-transformers)."""
    mask_expanded = np.expand_dims(attention_mask, axis=-1)
    summed = np.sum(token_embeddings * mask_expanded, axis=1)
    counts = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-9, a_max=None)


def encode_transformers(
    texts: Iterable[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    max_length: int,
) -> np.ndarray:
    encoded = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**encoded)
    pooled = mean_pool(
        outputs.last_hidden_state.cpu().numpy(),
        encoded["attention_mask"].cpu().numpy(),
    )
    return l2_normalize(pooled)


def encode_onnx(
    texts: Iterable[str],
    tokenizer: AutoTokenizer,
    session: ort.InferenceSession,
    max_length: int,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    for text in texts:
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
            "token_type_ids": encoded["token_type_ids"].astype(np.int64),
        }
        outputs = session.run(None, inputs)
        token_embeddings = outputs[0]  # last_hidden_state
        sentence_embedding = mean_pool(token_embeddings, inputs["attention_mask"])
        embeddings.append(sentence_embedding[0])
    return l2_normalize(np.stack(embeddings, axis=0))


def compare_embeddings(
    a: np.ndarray, b: np.ndarray, label_a: str, label_b: str
) -> None:
    print("\n" + "=" * 70)
    print(f"{label_a} vs {label_b}")
    print("=" * 70)

    cosine_sims = []
    for i in range(len(a)):
        cosine = float(np.dot(a[i], b[i]))
        cosine_sims.append(cosine)
        euclid = float(np.linalg.norm(a[i] - b[i]))
        mse = float(np.mean((a[i] - b[i]) ** 2))
        mae = float(np.mean(np.abs(a[i] - b[i])))
        print(f"Sentence {i+1}: cosine={cosine:.6f} euclid={euclid:.6f} mse={mse:.8f} mae={mae:.8f}")

    print("-" * 70)
    print(
        f"Average cosine: {np.mean(cosine_sims):.6f}  "
        f"min: {np.min(cosine_sims):.6f}  max: {np.max(cosine_sims):.6f}"
    )
    if np.mean(cosine_sims) > 0.99:
        verdict = "[OK] Embeddings match (cos > 0.99)"
    elif np.mean(cosine_sims) > 0.80:
        verdict = "[WARNING] Embeddings somewhat close (cos > 0.80)"
    else:
        verdict = "[ERROR] Embeddings diverge (cos <= 0.80)"
    print(verdict)


def find_webnn_files(model_dir: Path) -> tuple[Path, Path, Path]:
    webnn_file = model_dir / "model.webnn"
    weights_file = model_dir / "model.weights"
    manifest_file = model_dir / "manifest.json"
    for f in (webnn_file, weights_file, manifest_file):
        if not f.exists():
            raise FileNotFoundError(f"Missing required file: {f}")
    return webnn_file, weights_file, manifest_file


def encode_webnn(
    texts: Iterable[str],
    tokenizer: AutoTokenizer,
    model_dir: Path,
    max_length: int,
    export_onnx: Path | None = None,
) -> np.ndarray:
    # WebNN graph uses static [1, max_length] inputs; run one sentence at a time.
    ml = webnn.ML()
    context = ml.create_context(power_preference="default", accelerated=False)
    webnn_file, weights_file, manifest_file = find_webnn_files(model_dir)
    graph = webnn.MLGraph.load(
        str(webnn_file), manifest_path=str(manifest_file), weights_path=str(weights_file)
    )

    if export_onnx:
        try:
            context.convert_to_onnx(graph, str(export_onnx))
            print(f"[INFO] Exported WebNN graph to ONNX: {export_onnx}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] Failed to export WebNN graph to ONNX: {exc}")

    output_names = graph.get_output_names()
    if not output_names:
        raise RuntimeError("WebNN graph has no outputs")
    output_name = output_names[0]

    embeddings: list[np.ndarray] = []
    for text in texts:
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)
        token_type_ids = encoded.get("token_type_ids", np.zeros_like(input_ids)).astype(
            np.int64
        )
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        results = context.compute(graph, inputs)
        token_embeddings = results[output_name]
        sent_embedding = mean_pool(token_embeddings, attention_mask.astype(np.float32))
        embeddings.append(sent_embedding[0])

    return l2_normalize(np.stack(embeddings, axis=0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MiniLM embeddings across stages.")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=DEFAULT_STATIC_ONNX if DEFAULT_STATIC_ONNX.exists() else None,
        required=False,
        help="Path to original model-static.onnx.",
    )
    parser.add_argument(
        "--alt-onnx",
        type=Path,
        help="Optional path to an alternate ONNX (e.g., WebNN -> ONNX reconversion).",
    )
    parser.add_argument(
        "--webnn-dir",
        type=Path,
        default=DEFAULT_WEBNN_DIR if DEFAULT_WEBNN_DIR.exists() else None,
        help="Optional path to a WebNN model directory (model.webnn, model.weights, manifest.json).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Sequence length used in the static ONNX export.",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default=HF_MODEL_ID,
        help="Hugging Face model id to use as reference (default: sentence-transformers/all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--hf-max-length",
        type=int,
        default=128,
        help="Max sequence length for the HF reference encode (defaults to match ONNX/WebNN).",
    )
    parser.add_argument(
        "--export-webnn-onnx",
        type=Path,
        help="If set, export loaded WebNN graph to this ONNX path before running comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.onnx is None:
        raise SystemExit("Set --onnx to the original model-static.onnx path.")
    if not args.onnx.exists():
        raise SystemExit(f"ONNX file not found: {args.onnx}")

    test_sentences = [
        "This is a sample sentence to encode",
        "The cat sits on the mat",
        "A feline rests on the carpet",
        "The weather is sunny today",
        "Python is a programming language",
    ]

    print("=" * 70)
    print("MiniLM embedding sanity check")
    print("=" * 70)
    print(f"ONNX path: {args.onnx}")
    if args.alt_onnx:
        print(f"Alt ONNX: {args.alt_onnx}")

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id)
    model = AutoModel.from_pretrained(args.hf_model_id)

    print("\n[INFO] Encoding with transformers (reference)")
    ref_embeddings = encode_transformers(test_sentences, tokenizer, model, args.hf_max_length)

    print("\n[INFO] Encoding with original ONNX")
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    onnx_embeddings = encode_onnx(test_sentences, tokenizer, sess, args.max_length)

    compare_embeddings(ref_embeddings, onnx_embeddings, "Transformers", "ONNX (original export)")

    if args.alt_onnx:
        if not args.alt_onnx.exists():
            raise SystemExit(f"Alt ONNX not found: {args.alt_onnx}")
        print("\n[INFO] Encoding with alternate ONNX")
        alt_sess = ort.InferenceSession(str(args.alt_onnx), providers=["CPUExecutionProvider"])
        alt_embeddings = encode_onnx(test_sentences, tokenizer, alt_sess, args.max_length)
        compare_embeddings(ref_embeddings, alt_embeddings, "Transformers", "ONNX (alternate)")
        compare_embeddings(onnx_embeddings, alt_embeddings, "ONNX (original)", "ONNX (alternate)")

    if args.webnn_dir:
        if not args.webnn_dir.exists():
            raise SystemExit(f"WebNN dir not found: {args.webnn_dir}")
        print("\n[INFO] Encoding with WebNN")
        webnn_embeddings = encode_webnn(
            test_sentences,
            tokenizer,
            args.webnn_dir,
            args.max_length,
            export_onnx=args.export_webnn_onnx,
        )
        compare_embeddings(ref_embeddings, webnn_embeddings, "Transformers", "WebNN")
        compare_embeddings(onnx_embeddings, webnn_embeddings, "ONNX (original)", "WebNN")

    if args.export_webnn_onnx and args.export_webnn_onnx.exists():
        print("\n[INFO] Encoding with exported WebNN->ONNX")
        exported_sess = ort.InferenceSession(str(args.export_webnn_onnx), providers=["CPUExecutionProvider"])
        exported_embeddings = encode_onnx(test_sentences, tokenizer, exported_sess, args.max_length)
        compare_embeddings(ref_embeddings, exported_embeddings, "Transformers", "WebNN->ONNX export")
        compare_embeddings(onnx_embeddings, exported_embeddings, "ONNX (original)", "WebNN->ONNX export")
        compare_embeddings(webnn_embeddings, exported_embeddings, "WebNN", "WebNN->ONNX export")


if __name__ == "__main__":
    main()
