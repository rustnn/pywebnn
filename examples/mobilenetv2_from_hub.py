#!/usr/bin/env python3
"""
MobileNetV2 Image Classification - Loading from Hugging Face Hub
=================================================================

This demo loads a pre-trained MobileNetV2 model directly from Hugging Face Hub
and runs inference on an image.

The model is downloaded from the Hub and cached locally:
- Graph structure (.webnn file)
- Binary weights (.weights file)
- Weight manifest (manifest.json)

Usage:
    python examples/mobilenetv2_from_hub.py <image_path> [--backend cpu|gpu|coreml]

Example:
    python examples/mobilenetv2_from_hub.py examples/images/test.jpg --backend cpu
    python examples/mobilenetv2_from_hub.py cat.jpg --model-id username/my-model
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import webnn


# Load ImageNet labels
def load_imagenet_labels():
    labels_file = Path(__file__).parent / "imagenet_classes.txt"
    if not labels_file.exists():
        # Download from GitHub if not found
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        print(f"   [DOWNLOAD] ImageNet labels from GitHub...")
        urllib.request.urlretrieve(url, labels_file)
        print(f"   [OK] Labels downloaded")

    with open(labels_file) as f:
        return [line.strip() for line in f]


IMAGENET_CLASSES = load_imagenet_labels()


def preprocess_image(image_path):
    """Preprocess image for MobileNetV2."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.BILINEAR)

    img_array = np.array(img, dtype=np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    # HWC to CHW and add batch
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def main():
    parser = argparse.ArgumentParser(
        description="MobileNetV2 Image Classification (Hugging Face Hub)"
    )
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument(
        "--backend",
        choices=["cpu", "gpu", "coreml"],
        default="cpu",
        help="Backend to use for inference"
    )
    parser.add_argument(
        "--model-id",
        default="tarekziade/mobilenet-webnn",
        help="Hugging Face model ID (default: tarekziade/mobilenet-webnn)"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if cached"
    )
    args = parser.parse_args()

    # Verify image exists
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Backend settings
    if args.backend == "cpu":
        accelerated, power, backend_name = False, "default", "ONNX CPU"
    elif args.backend == "gpu":
        accelerated, power, backend_name = True, "high-performance", "ONNX GPU"
    else:
        accelerated, power, backend_name = True, "high-performance", "CoreML (Neural Engine)"

    print("=" * 70)
    print("MobileNetV2 Image Classification (Hugging Face Hub)")
    print("=" * 70)
    print(f"Image: {image_path}")
    print(f"Model: {args.model_id}")
    print(f"Backend: {backend_name}")
    print()

    # Download model from Hub
    print("Downloading model from Hugging Face Hub...")
    start_time = time.time()
    hub = webnn.Hub()
    model_files = hub.download_model(args.model_id, force=args.force_download)
    download_time = (time.time() - start_time) * 1000

    print(f"   [OK] Model downloaded ({download_time:.2f}ms)")
    print(f"   - Graph: {Path(model_files['graph']).name}")
    print()

    # Load graph from downloaded files
    print("Loading graph from downloaded files...")
    start_time = time.time()
    graph = webnn.MLGraph.load(
        model_files['graph'],
        manifest_path=model_files['manifest'],
        weights_path=model_files['weights']
    )
    load_time = (time.time() - start_time) * 1000

    print(f"   [OK] Graph loaded successfully ({load_time:.2f}ms)")
    print(f"   - Operand count: {graph.operand_count}")
    print(f"   - Operation count: {graph.operation_count}")
    print(f"   - Inputs: {graph.get_input_names()}")
    print(f"   - Outputs: {graph.get_output_names()}")
    print()

    # Preprocess image
    print("Preprocessing image...")
    start_time = time.time()
    input_data = preprocess_image(image_path)
    prep_time = (time.time() - start_time) * 1000
    print(f"   [OK] Preprocessed to {input_data.shape} ({prep_time:.2f}ms)")
    print()

    # Create context
    print("Creating WebNN context...")
    ml = webnn.ML()
    context = ml.create_context(power_preference=power, accelerated=accelerated)
    print(f"   [OK] Context created (accelerated={context.accelerated})")
    print()

    # Run inference
    print("Running inference...")
    start_time = time.time()
    results = context.compute(graph, {"input": input_data})
    inf_time = (time.time() - start_time) * 1000
    print(f"   [OK] Inference complete ({inf_time:.2f}ms)")
    print()

    # Get predictions
    output_probs = results["output"][0]
    top_indices = np.argsort(output_probs)[-5:][::-1]

    print("Top 5 Predictions:")
    print("-" * 70)
    for i, idx in enumerate(top_indices, 1):
        class_name = IMAGENET_CLASSES[idx]
        confidence = output_probs[idx] * 100
        print(f"   {i}. {class_name:50s} {confidence:6.2f}%")
    print()

    print("=" * 70)
    print("Performance Summary:")
    print(f"  - Model Download: {download_time:.2f}ms")
    print(f"  - Graph Load:     {load_time:.2f}ms")
    print(f"  - Preprocessing:  {prep_time:.2f}ms")
    print(f"  - Inference:      {inf_time:.2f}ms")
    print(f"  - Total Time:     {download_time + load_time + prep_time + inf_time:.2f}ms")
    print("=" * 70)
    print()

    # Show cache info
    cached_models = hub.list_cached_models()
    if cached_models:
        print("[INFO] Cached models:")
        for model in cached_models:
            print(f"   - {model}")

    print()
    print("[OK] MobileNetV2 classification complete!")
    print(f"     Model loaded from Hugging Face Hub: {args.model_id}")
    print(f"     Files cached at: {hub.cache_dir}")


if __name__ == "__main__":
    main()
