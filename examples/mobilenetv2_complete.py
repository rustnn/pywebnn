#!/usr/bin/env python3
"""
Complete MobileNetV2 with WebNN - Full 106-layer Implementation
================================================================

This implements the complete MobileNetV2 architecture using WebNN operations,
exactly like the JavaScript demos, with all 106 pretrained weight layers.

Architecture:
- Initial conv block (32 channels)
- 17 inverted residual blocks
- Final conv (1280 channels)
- Global average pool
- Classifier (1000 classes)

All weights loaded from webmachinelearning/test-data repository.
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
    with open(labels_file) as f:
        return [line.strip() for line in f]

IMAGENET_CLASSES = load_imagenet_labels()


def load_weights(weights_dir="mobilenetv2_weights"):
    """Load all 106 MobileNetV2 pretrained weights."""
    print("Loading all pretrained MobileNetV2 weights...")

    script_dir = Path(__file__).parent
    weights_path = script_dir / weights_dir

    weights = {}

    # Conv layer indices from WebNN test-data
    conv_layers = [0, 2, 4, 5, 7, 9, 10, 12, 14, 16, 18, 20, 21, 23, 25, 27,
                   29, 31, 33, 35, 37, 38, 40, 42, 44, 46, 48, 50, 52, 54,
                   56, 58, 60, 61, 63, 65, 67, 69, 71, 73, 75, 77, 78, 80,
                   82, 84, 86, 88, 90, 92, 94, 95]

    for idx in conv_layers:
        weight_file = weights_path / f"conv_{idx}_weight.npy"
        bias_file = weights_path / f"conv_{idx}_bias.npy"

        if weight_file.exists() and bias_file.exists():
            weights[f'conv_{idx}_weight'] = np.load(weight_file)
            weights[f'conv_{idx}_bias'] = np.load(bias_file)

    # Load classifier
    fc_weight_file = weights_path / "gemm_104_weight.npy"
    fc_bias_file = weights_path / "gemm_104_bias.npy"

    if fc_weight_file.exists() and fc_bias_file.exists():
        weights['fc_weight'] = np.load(fc_weight_file)
        weights['fc_bias'] = np.load(fc_bias_file)

    print(f"   ✓ Loaded {len(weights)} weight tensors")
    return weights


def build_complete_mobilenetv2(builder, weights):
    """
    Build complete MobileNetV2 architecture with all layers.

    MobileNetV2 structure:
    - Initial conv: 3x3, stride 2, 32 filters
    - 17 Inverted Residual blocks
    - Final conv: 1x1, 1280 filters
    - Global average pool
    - FC classifier: 1000 classes
    """
    print("Building complete MobileNetV2 graph...")

    # Input: (1, 3, 224, 224)
    x = builder.input("input", [1, 3, 224, 224], "float32")

    # Layer numbering follows WebNN test-data
    layer_idx = 0

    # Initial conv block: conv_0 (stride 2)
    print(f"   Layer {layer_idx}: Initial conv 3->32")
    w = builder.constant(weights['conv_0_weight'], list(weights['conv_0_weight'].shape), "float32")
    b = builder.constant(weights['conv_0_bias'].reshape(1, -1, 1, 1), [1, 32, 1, 1], "float32")
    x = builder.conv2d(x, w, [2, 2], None, [1, 1, 1, 1])
    x = builder.add(x, b)
    x = builder.clamp(x, 0.0, 6.0)  # ReLU6
    layer_idx += 1

    # MobileNetV2 Inverted Residual Blocks with explicit weight indices
    # Structure: (expansion, out_channels, stride, weight_indices)
    # weight_indices = [expansion_conv, depthwise_conv, projection_conv] or [depthwise, projection] if no expansion
    block_specs = [
        # Block 0: 32 -> 16, stride 1, no expansion (t=1)
        (1, 16, 1, [2, 4]),
        # Block 1-2: 16 -> 24, stride 2 then 1, expansion 6
        (6, 24, 2, [5, 7, 9]),
        (6, 24, 1, [10, 12, 14]),
        # Block 3-5: 24 -> 32, stride 2 then 1, 1, expansion 6
        (6, 32, 2, [16, 18, 20]),
        (6, 32, 1, [21, 23, 25]),
        (6, 32, 1, [27, 29, 31]),
        # Block 6-9: 32 -> 64, stride 2 then 1, 1, 1, expansion 6
        (6, 64, 2, [33, 35, 37]),
        (6, 64, 1, [38, 40, 42]),
        (6, 64, 1, [44, 46, 48]),
        (6, 64, 1, [50, 52, 54]),
        # Block 10-12: 64 -> 96, stride 1, 1, 1, expansion 6
        (6, 96, 1, [56, 58, 60]),
        (6, 96, 1, [61, 63, 65]),
        (6, 96, 1, [67, 69, 71]),
        # Block 13-15: 96 -> 160, stride 2 then 1, 1, expansion 6
        (6, 160, 2, [73, 75, 77]),
        (6, 160, 1, [78, 80, 82]),
        (6, 160, 1, [84, 86, 88]),
        # Block 16: 160 -> 320, stride 1, expansion 6
        (6, 320, 1, [90, 92, 94]),
    ]

    current_channels = 32

    for block_idx, (expansion, out_channels, block_stride, weight_indices) in enumerate(block_specs):
        use_residual = (block_stride == 1 and current_channels == out_channels)

        print(f"   Block {block_idx}: {current_channels}->{out_channels} (stride={block_stride}, expansion={expansion})")

        identity = x

        if expansion == 1:
            # No expansion, just depthwise + projection
            dw_idx, proj_idx = weight_indices
            exp_channels = current_channels

            # Depthwise conv
            w = builder.constant(weights[f'conv_{dw_idx}_weight'],
                               list(weights[f'conv_{dw_idx}_weight'].shape), "float32")
            b = builder.constant(weights[f'conv_{dw_idx}_bias'].reshape(1, -1, 1, 1),
                               [1, exp_channels, 1, 1], "float32")
            x = builder.conv2d(x, w, [block_stride, block_stride], None, [1, 1, 1, 1], exp_channels)
            x = builder.add(x, b)
            x = builder.clamp(x, 0.0, 6.0)

            # Projection
            w = builder.constant(weights[f'conv_{proj_idx}_weight'],
                               list(weights[f'conv_{proj_idx}_weight'].shape), "float32")
            b = builder.constant(weights[f'conv_{proj_idx}_bias'].reshape(1, -1, 1, 1),
                               [1, out_channels, 1, 1], "float32")
            x = builder.conv2d(x, w, None, None, None)
            x = builder.add(x, b)

        else:
            # Expansion -> Depthwise -> Projection
            exp_idx, dw_idx, proj_idx = weight_indices
            exp_channels = current_channels * expansion

            # Expansion
            w = builder.constant(weights[f'conv_{exp_idx}_weight'],
                               list(weights[f'conv_{exp_idx}_weight'].shape), "float32")
            b = builder.constant(weights[f'conv_{exp_idx}_bias'].reshape(1, -1, 1, 1),
                               [1, exp_channels, 1, 1], "float32")
            x = builder.conv2d(x, w, None, None, None)
            x = builder.add(x, b)
            x = builder.clamp(x, 0.0, 6.0)

            # Depthwise
            w = builder.constant(weights[f'conv_{dw_idx}_weight'],
                               list(weights[f'conv_{dw_idx}_weight'].shape), "float32")
            b = builder.constant(weights[f'conv_{dw_idx}_bias'].reshape(1, -1, 1, 1),
                               [1, exp_channels, 1, 1], "float32")
            x = builder.conv2d(x, w, [block_stride, block_stride], None, [1, 1, 1, 1], exp_channels)
            x = builder.add(x, b)
            x = builder.clamp(x, 0.0, 6.0)

            # Projection
            w = builder.constant(weights[f'conv_{proj_idx}_weight'],
                               list(weights[f'conv_{proj_idx}_weight'].shape), "float32")
            b = builder.constant(weights[f'conv_{proj_idx}_bias'].reshape(1, -1, 1, 1),
                               [1, out_channels, 1, 1], "float32")
            x = builder.conv2d(x, w, None, None, None)
            x = builder.add(x, b)

        # Residual connection
        if use_residual:
            x = builder.add(x, identity)

        current_channels = out_channels

    # Final conv: conv_95 (1x1, 1280 filters)
    print(f"   Layer final: Conv {current_channels}->1280")
    w = builder.constant(weights['conv_95_weight'], list(weights['conv_95_weight'].shape), "float32")
    b = builder.constant(weights['conv_95_bias'].reshape(1, -1, 1, 1), [1, 1280, 1, 1], "float32")
    x = builder.conv2d(x, w, None, None, None)
    x = builder.add(x, b)
    x = builder.clamp(x, 0.0, 6.0)

    # Global average pooling
    print("   Layer: Global average pool")
    x = builder.global_average_pool(x)
    x = builder.reshape(x, [1, 1280])

    # Classifier
    print("   Layer: Classifier 1280->1000")
    fc_w = builder.constant(weights['fc_weight'], list(weights['fc_weight'].shape), "float32")
    fc_b = builder.constant(weights['fc_bias'].reshape(1, -1), [1, 1000], "float32")
    x = builder.gemm(x, fc_w, b_transpose=True)
    x = builder.add(x, fc_b)

    # Softmax
    output = builder.softmax(x)

    print("   ✓ Complete MobileNetV2 graph built!")
    return output


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
    parser = argparse.ArgumentParser(description="Complete MobileNetV2 with WebNN")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--backend", choices=["cpu", "gpu", "coreml"], default="cpu")
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
        accelerated, power, backend_name = True, "high-performance", "CoreML (Neural Engine)"

    print("=" * 70)
    print("Complete MobileNetV2 Image Classification with WebNN")
    print("=" * 70)
    print(f"Image: {args.image_path}")
    print(f"Backend: {backend_name}")
    print()

    # Load weights
    start_time = time.time()
    weights = load_weights()
    load_time = (time.time() - start_time) * 1000
    print(f"   Weight load time: {load_time:.2f}ms")
    print()

    # Preprocess image
    print("Preprocessing image...")
    start_time = time.time()
    input_data = preprocess_image(args.image_path)
    prep_time = (time.time() - start_time) * 1000
    print(f"   ✓ Preprocessed to {input_data.shape} ({prep_time:.2f}ms)")
    print()

    # Create context
    print("Creating WebNN context...")
    ml = webnn.ML()
    context = ml.create_context(power_preference=power, accelerated=accelerated)
    print(f"   ✓ Context created (accelerated={context.accelerated})")
    print()

    # Build graph
    start_time = time.time()
    builder = context.create_graph_builder()
    output = build_complete_mobilenetv2(builder, weights)
    graph = builder.build({"output": output})
    build_time = (time.time() - start_time) * 1000
    print(f"   Graph build time: {build_time:.2f}ms")
    print()

    # Run inference
    print("Running inference...")
    start_time = time.time()
    results = context.compute(graph, {"input": input_data})
    inf_time = (time.time() - start_time) * 1000
    print(f"   ✓ Inference complete ({inf_time:.2f}ms)")
    print()

    # Get predictions
    output_probs = results["output"][0]
    top_indices = np.argsort(output_probs)[-5:][::-1]

    print("Top 5 Predictions (Real ImageNet Labels):")
    print("-" * 70)
    for i, idx in enumerate(top_indices, 1):
        class_name = IMAGENET_CLASSES[idx]
        confidence = output_probs[idx] * 100
        print(f"   {i}. {class_name:50s} {confidence:6.2f}%")
    print()

    print("=" * 70)
    print("Performance Summary:")
    print(f"  - Weight Load:   {load_time:.2f}ms")
    print(f"  - Preprocessing: {prep_time:.2f}ms")
    print(f"  - Graph Build:   {build_time:.2f}ms")
    print(f"  - Inference:     {inf_time:.2f}ms")
    print("=" * 70)
    print()
    print("✓ Complete MobileNetV2 with all 106 pretrained layers!")
    print("  Architecture built using WebNN operations (conv2d, add, clamp, etc.)")
    print("  Just like the JavaScript WebNN demos!")


if __name__ == "__main__":
    main()
