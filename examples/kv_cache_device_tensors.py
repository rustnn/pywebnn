#!/usr/bin/env python3
"""KV-cache updates with persistent native tensors.

The historical filename is retained. Compare explicit host round trips with
ping-pong tensor bindings; actual storage and transfers are controlled by RustNN.
"""

import sys
import time
import numpy as np

try:
    import webnn
except ImportError:
    print("Error: webnn package not found")
    print("Build and install with: make python-dev")
    sys.exit(1)


def build_decoder_graph(ctx, batch_size=1, seq_len=128, hidden_dim=256, num_heads=4):
    """
    Build a simplified KV cache update graph.

    The graph takes:
    - past_key: [batch, num_heads, seq_len, head_dim] - cached keys
    - past_value: [batch, num_heads, seq_len, head_dim] - cached values

    Returns:
    - present_key: [batch, num_heads, seq_len, head_dim] - updated keys
    - present_value: [batch, num_heads, seq_len, head_dim] - updated values
    """
    builder = ctx.create_graph_builder()
    head_dim = hidden_dim // num_heads

    # Inputs
    past_key = builder.input(
        "past_key", [batch_size, num_heads, seq_len, head_dim], "float32"
    )
    past_value = builder.input(
        "past_value", [batch_size, num_heads, seq_len, head_dim], "float32"
    )

    # For demo purposes: simulate KV cache update with small increment
    # present_key = past_key + 0.01
    # present_value = past_value + 0.01
    scale = builder.constant(
        np.full([batch_size, num_heads, seq_len, head_dim], 0.01, dtype=np.float32)
    )
    present_key = builder.add(past_key, scale)
    present_value = builder.add(past_value, scale)

    # Build graph
    graph = builder.build(
        {
            "present_key": present_key,
            "present_value": present_value,
        }
    )

    return graph


def run_decode(ctx, graph, num_steps=50, *, host_round_trip=False):
    """Compare host round trips with reuse of native ping-pong tensors.

    RustNN chooses storage and transfers. This example makes no zero-copy or
    device-residency guarantee.
    """
    shape = [1, 4, 128, 64]
    past = {
        name: ctx.create_tensor(shape, "float32", readable=True, writable=True)
        for name in ("past_key", "past_value")
    }
    present = {
        name: ctx.create_tensor(shape, "float32", readable=True, writable=True)
        for name in ("present_key", "present_value")
    }
    for tensor in past.values():
        ctx.write_tensor(tensor, np.zeros(shape, dtype=np.float32))

    start = time.perf_counter()
    for _ in range(num_steps):
        ctx.dispatch(graph, past, present)
        if host_round_trip:
            ctx.write_tensor(past["past_key"], ctx.read_tensor(present["present_key"]))
            ctx.write_tensor(
                past["past_value"], ctx.read_tensor(present["present_value"])
            )
        else:
            past["past_key"], present["present_key"] = (
                present["present_key"],
                past["past_key"],
            )
            past["past_value"], present["present_value"] = (
                present["present_value"],
                past["past_value"],
            )
    elapsed = time.perf_counter() - start
    final_key = ctx.read_tensor(past["past_key"])
    np.testing.assert_allclose(final_key, num_steps * 0.01, rtol=1e-5, atol=1e-6)
    label = "Host round trips" if host_round_trip else "Persistent ping-pong tensors"
    print(f"{label}: {elapsed:.3f}s ({num_steps / elapsed:.1f} steps/sec)")
    return final_key


def main():
    ctx = webnn.ML().create_context(backend="onnx", accelerated=False)
    graph = build_decoder_graph(ctx)
    reference = run_decode(ctx, graph, host_round_trip=True)
    result = run_decode(ctx, graph)
    np.testing.assert_allclose(result, reference)


if __name__ == "__main__":
    main()
