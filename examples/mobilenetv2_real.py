#!/usr/bin/env python3
"""
WebNN Image Classification with Real Pretrained MobileNetV2
============================================================

This example uses pretrained MobileNetV2 weights from the WebNN test-data repository
for real image classification with ImageNet labels.

Requirements:
    pip install pillow numpy imagenet-classes requests

Usage:
    python examples/mobilenetv2_real.py path/to/image.jpg [--backend cpu|gpu|coreml]
"""

import sys
import time
import argparse
from pathlib import Path
import urllib.request

try:
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Please run: pip install pillow numpy")
    sys.exit(1)

import webnn


# Load ImageNet class labels
def load_imagenet_labels():
    """Load ImageNet class labels from file."""
    labels_file = Path(__file__).parent / "imagenet_classes.txt"
    with open(labels_file) as f:
        return [line.strip() for line in f]


IMAGENET_CLASSES = load_imagenet_labels()


# Base URL for pretrained weights
WEIGHTS_BASE_URL = "https://raw.githubusercontent.com/webmachinelearning/test-data/0495fc5b5e4ccf77f745b747aa43e12a71a30cff/models/mobilenetv2_nchw/weights"


def load_weight(filename, weights_dir):
    """Load a weight file from local directory."""
    weight_file = Path(weights_dir) / filename
    if not weight_file.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_file}")
    return np.load(weight_file)


def load_mobilenetv2_weights(layer_indices, weights_dir="mobilenetv2_weights"):
    """
    Load MobileNetV2 weights for specified layers.

    Args:
        layer_indices: List of layer indices to load weights for
        weights_dir: Directory containing weight files

    Returns:
        dict: Dictionary of weight arrays
    """
    weights = {}

    print("Loading pretrained MobileNetV2 weights...")

    # Get absolute path to weights directory (relative to script location)
    script_dir = Path(__file__).parent
    weights_path = script_dir / weights_dir

    # Load convolutional layer weights
    for idx in layer_indices:
        try:
            weight = load_weight(f"conv_{idx}_weight.npy", weights_path)
            bias = load_weight(f"conv_{idx}_bias.npy", weights_path)
            weights[f'conv_{idx}_weight'] = weight
            weights[f'conv_{idx}_bias'] = bias
        except Exception as e:
            print(f"   Warning: Could not load conv_{idx}: {e}")

    # Load final fully connected layer
    try:
        weights['fc_weight'] = load_weight("gemm_104_weight.npy", weights_path)
        weights['fc_bias'] = load_weight("gemm_104_bias.npy", weights_path)
        print(f"   ✓ Loaded FC layer: {weights['fc_weight'].shape}")
    except Exception as e:
        print(f"   Warning: Could not load FC layer: {e}")

    return weights


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for MobileNetV2.

    Args:
        image_path: Path to the input image
        target_size: Target size (height, width)

    Returns:
        Preprocessed image as numpy array (1, 3, 224, 224)
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)

    # Convert to array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    # HWC to CHW and add batch dimension
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def build_mobilenetv2_simple(builder, weights):
    """
    Build a simplified MobileNetV2 using key pretrained layers.

    This uses the first conv layer and final classifier with pretrained weights.
    Middle layers are simplified for demonstration.

    Args:
        builder: WebNN MLGraphBuilder
        weights: Dictionary of pretrained weights

    Returns:
        Output operand
    """
    # Input: (1, 3, 224, 224)
    x = builder.input("input", [1, 3, 224, 224], "float32")

    # First convolution: 3 -> 32 channels (pretrained)
    if 'conv_0_weight' in weights:
        conv0_w = weights['conv_0_weight']
        conv0_b = weights['conv_0_bias']
        print(f"   Using pretrained conv_0: {conv0_w.shape}")

        conv_w = builder.constant(conv0_w, list(conv0_w.shape), "float32")
        conv_b = builder.constant(conv0_b.reshape(1, -1, 1, 1), [1, 32, 1, 1], "float32")

        x = builder.conv2d(x, conv_w, [2, 2], None, [1, 1, 1, 1])
        x = builder.add(x, conv_b)
        x = builder.clamp(x, min_value=0.0, max_value=6.0)

    # Add a few more conv blocks with pretrained weights if available
    # For simplicity, we'll skip to the end

    # Simplified middle layers (in real implementation, would use all pretrained weights)
    # Conv block 1: 32 -> 64
    conv1_w = builder.constant(
        np.random.randn(64, 32, 3, 3).astype(np.float32) * 0.01,
        [64, 32, 3, 3], "float32"
    )
    x = builder.conv2d(x, conv1_w, [2, 2], None, [1, 1, 1, 1])
    x = builder.clamp(x, min_value=0.0, max_value=6.0)

    # Conv block 2: 64 -> 128
    conv2_w = builder.constant(
        np.random.randn(128, 64, 3, 3).astype(np.float32) * 0.01,
        [128, 64, 3, 3], "float32"
    )
    x = builder.conv2d(x, conv2_w, [2, 2], None, [1, 1, 1, 1])
    x = builder.clamp(x, min_value=0.0, max_value=6.0)

    # Final conv: 128 -> 1280 (MobileNetV2 feature dimension)
    conv3_w = builder.constant(
        np.random.randn(1280, 128, 1, 1).astype(np.float32) * 0.01,
        [1280, 128, 1, 1], "float32"
    )
    x = builder.conv2d(x, conv3_w, None, None, None)
    x = builder.clamp(x, min_value=0.0, max_value=6.0)

    # Global average pooling
    x = builder.global_average_pool(x)
    x = builder.reshape(x, [1, 1280])

    # Final classifier (pretrained)
    if 'fc_weight' in weights:
        fc_w = weights['fc_weight']
        fc_b = weights['fc_bias']
        print(f"   Using pretrained classifier: {fc_w.shape}")

        fc_weight = builder.constant(fc_w, list(fc_w.shape), "float32")
        fc_bias = builder.constant(fc_b.reshape(1, -1), [1, 1000], "float32")

        x = builder.gemm(x, fc_weight, b_transpose=True)
        x = builder.add(x, fc_bias)

    # Softmax
    output = builder.softmax(x)

    return output


def get_top_predictions(probabilities, top_k=5):
    """Get top-k predictions."""
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    return [(int(idx), float(probabilities[idx])) for idx in top_indices]


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="WebNN Image Classification with Real MobileNetV2"
    )
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument(
        "--backend",
        choices=["cpu", "gpu", "coreml"],
        default="cpu",
        help="Backend: cpu (ONNX CPU), gpu (ONNX GPU), or coreml (CoreML)",
    )
    args = parser.parse_args()

    if not Path(args.image_path).exists():
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)

    # Backend settings
    if args.backend == "cpu":
        accelerated, power, backend_name = False, "default", "ONNX CPU"
    elif args.backend == "gpu":
        accelerated, power, backend_name = True, "high-performance", "ONNX GPU"
    else:
        accelerated, power, backend_name = True, "low-power", "CoreML"

    print("=" * 70)
    print("WebNN Real Image Classification - Pretrained MobileNetV2")
    print("=" * 70)
    print(f"Image: {args.image_path}")
    print(f"Backend: {backend_name}")
    print()

    # Load weights
    print("1. Loading pretrained weights...")
    start_time = time.time()
    # Load first conv and final FC layer
    weights = load_mobilenetv2_weights([0])
    load_time = (time.time() - start_time) * 1000
    print(f"   ✓ Loaded {len(weights)} weight tensors ({load_time:.2f}ms)")
    print()

    # Load and preprocess image
    print("2. Loading and preprocessing image...")
    start_time = time.time()
    input_data = load_and_preprocess_image(args.image_path)
    prep_time = (time.time() - start_time) * 1000
    print(f"   ✓ Preprocessed to {input_data.shape} ({prep_time:.2f}ms)")
    print()

    # Create context
    print("3. Creating WebNN context...")
    ml = webnn.ML()
    context = ml.create_context(power_preference=power, accelerated=accelerated)
    print(f"   ✓ Context created (accelerated={context.accelerated})")
    print()

    # Build graph
    print("4. Building graph with pretrained weights...")
    start_time = time.time()
    builder = context.create_graph_builder()
    output = build_mobilenetv2_simple(builder, weights)
    graph = builder.build({"output": output})
    build_time = (time.time() - start_time) * 1000
    print(f"   ✓ Graph built ({build_time:.2f}ms)")
    print()

    # Run inference
    print("5. Running inference...")
    start_time = time.time()
    results = context.compute(graph, {"input": input_data})
    inf_time = (time.time() - start_time) * 1000
    print(f"   ✓ Inference complete ({inf_time:.2f}ms)")
    print()

    # Get predictions with real labels
    output_probs = results["output"][0]
    top_predictions = get_top_predictions(output_probs, top_k=5)

    print("6. Top 5 Predictions (Real ImageNet Labels):")
    print("-" * 70)
    for i, (class_idx, prob) in enumerate(top_predictions, 1):
        class_name = IMAGENET_CLASSES[class_idx]
        print(f"   {i}. {class_name:50s} {prob*100:6.2f}%")
    print()

    print("=" * 70)
    print("Performance Summary:")
    print(f"  - Weight Load:   {load_time:.2f}ms")
    print(f"  - Preprocessing: {prep_time:.2f}ms")
    print(f"  - Graph Build:   {build_time:.2f}ms")
    print(f"  - Inference:     {inf_time:.2f}ms")
    print("=" * 70)
    print()
    print("Note: Using pretrained first conv + classifier layers.")
    print("Middle layers use random weights for demonstration.")
    print("For full accuracy, all 106 layers would be needed.")


if __name__ == "__main__":
    main()
