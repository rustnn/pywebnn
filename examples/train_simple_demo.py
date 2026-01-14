#!/usr/bin/env python3
"""
Simple Working Training Demo
=============================

This is a MINIMAL demo that actually shows training working:
- Character-level (a-z, space, punctuation)
- Very simple model (no attention)
- Proper gradient updates
- Loss actually decreases!

Goal: Demonstrate that the training workflow works, even if simplified.

Usage:
    python train_simple_demo.py --phrase "hello world" --epochs 100
"""

import argparse
import json
import numpy as np
from pathlib import Path


# Character-level vocabulary
CHARS = " abcdefghijklmnopqrstuvwxyz.,!?-"
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}


class SimpleCharModel:
    """
    Ultra-simple character-level model.
    Just: embedding -> hidden layer -> output
    No attention, no transformer complexity.
    """

    def __init__(self, vocab_size=len(CHARS), hidden_size=32):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Initialize small random weights
        np.random.seed(42)
        self.embed = np.random.randn(vocab_size, hidden_size) * 0.1
        self.W1 = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, vocab_size) * 0.1
        self.b2 = np.zeros(vocab_size)

    def forward(self, x):
        """
        Forward pass.
        x: sequence of char indices (seq_len,)
        returns: probabilities for next char (vocab_size,)
        """
        # Average embeddings of input sequence
        embedded = np.mean([self.embed[idx] for idx in x], axis=0)  # (hidden_size,)

        # Hidden layer with ReLU
        hidden = np.maximum(0, embedded @ self.W1 + self.b1)

        # Output layer
        logits = hidden @ self.W2 + self.b2

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        return probs, hidden, embedded

    def compute_loss(self, text):
        """Compute average loss on a text sequence."""
        total_loss = 0
        count = 0

        for i in range(1, len(text)):
            x = text[:i]
            target = text[i]

            probs, _, _ = self.forward(x)
            loss = -np.log(probs[target] + 1e-10)
            total_loss += loss
            count += 1

        return total_loss / count if count > 0 else 0

    def train_step(self, text, lr=0.1):
        """
        Train on a sequence using simplified backprop.
        Updates all parameters with proper gradients.
        """
        # Compute loss before
        loss_before = self.compute_loss(text)

        # Accumulate gradients
        grad_embed = np.zeros_like(self.embed)
        grad_W1 = np.zeros_like(self.W1)
        grad_b1 = np.zeros_like(self.b1)
        grad_W2 = np.zeros_like(self.W2)
        grad_b2 = np.zeros_like(self.b2)

        # For each position in the sequence
        for i in range(1, len(text)):
            x = text[:i]
            target = text[i]

            # Forward pass
            probs, hidden, embedded = self.forward(x)

            # Gradient of loss w.r.t. output (cross-entropy + softmax)
            grad_output = probs.copy()
            grad_output[target] -= 1

            # Backprop through output layer
            grad_W2 += np.outer(hidden, grad_output)
            grad_b2 += grad_output
            grad_hidden = grad_output @ self.W2.T

            # Backprop through ReLU
            grad_hidden[hidden <= 0] = 0

            # Backprop through hidden layer
            grad_W1 += np.outer(embedded, grad_hidden)
            grad_b1 += grad_hidden
            grad_embedded = grad_hidden @ self.W1.T

            # Backprop to embeddings (distribute to all input chars)
            for idx in x:
                grad_embed[idx] += grad_embedded / len(x)

        # Update parameters
        self.embed -= lr * grad_embed
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2

        return loss_before

    def generate(self, prompt, max_len=20):
        """Generate text from a prompt."""
        text = [CHAR_TO_IDX.get(c, 0) for c in prompt.lower() if c in CHAR_TO_IDX]

        for _ in range(max_len):
            if len(text) == 0:
                break
            probs, _, _ = self.forward(text)
            next_char = np.random.choice(self.vocab_size, p=probs)
            text.append(next_char)

        return ''.join([IDX_TO_CHAR.get(idx, '?') for idx in text])

    def save(self, filename):
        """Save weights to JSON."""
        weights = {
            'embed': self.embed.tolist(),
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.tolist(),
            'b2': self.b2.tolist(),
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size
        }
        with open(filename, 'w') as f:
            json.dump(weights, f)
        print(f"✓ Saved weights to {filename}")

    def load(self, filename):
        """Load weights from JSON."""
        with open(filename, 'r') as f:
            weights = json.load(f)
        self.embed = np.array(weights['embed'])
        self.W1 = np.array(weights['W1'])
        self.b1 = np.array(weights['b1'])
        self.W2 = np.array(weights['W2'])
        self.b2 = np.array(weights['b2'])
        print(f"✓ Loaded weights from {filename}")


def main():
    parser = argparse.ArgumentParser(description="Simple working training demo")
    parser.add_argument("--phrase", default="the quick brown fox jumps", help="Phrase to memorize")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--save", default="simple_model.json", help="Where to save weights")
    args = parser.parse_args()

    # Convert phrase to indices
    phrase_clean = args.phrase.lower()
    phrase_indices = [CHAR_TO_IDX.get(c, 0) for c in phrase_clean if c in CHAR_TO_IDX]

    if len(phrase_indices) < 2:
        print("Error: Phrase too short")
        return

    print("=" * 70)
    print("Simple Working Training Demo")
    print("=" * 70)
    print(f"Phrase: '{phrase_clean}'")
    print(f"Length: {len(phrase_indices)} chars")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print()

    # Initialize model
    model = SimpleCharModel()
    print(f"✓ Model initialized (vocab={model.vocab_size}, hidden={model.hidden_size})")
    print()

    # Test before training
    print("Before training:")
    print(f"  Loss: {model.compute_loss(phrase_indices):.4f}")
    print(f"  Generated: '{model.generate(phrase_clean[:3], max_len=len(phrase_clean))}'")
    print()

    # Training loop
    print("Training...")
    print("-" * 70)

    for epoch in range(args.epochs):
        loss = model.train_step(phrase_indices, lr=args.lr)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            generated = model.generate(phrase_clean[:3], max_len=len(phrase_clean))
            print(f"Epoch {epoch+1:3d}: Loss={loss:.4f} | Generated: '{generated}'")

    print("-" * 70)
    print()

    # Test after training
    final_loss = model.compute_loss(phrase_indices)
    final_gen = model.generate(phrase_clean[:3], max_len=len(phrase_clean))

    print("After training:")
    print(f"  Loss: {final_loss:.4f} (started at ~{np.log(model.vocab_size):.2f})")
    print(f"  Generated: '{final_gen}'")
    print()

    # Save model
    model.save(args.save)
    print()

    print("=" * 70)
    print("✓ Training complete!")
    print()
    print("What this demonstrates:")
    print("  • Loss decreases (model is learning)")
    print("  • Output becomes more like training phrase")
    print("  • Weights can be saved and loaded")
    print("  • Training workflow works end-to-end")
    print()
    print("Note: This is a toy model for demonstration. For real text")
    print("generation, use proper frameworks (PyTorch) and larger models.")
    print("=" * 70)


if __name__ == "__main__":
    main()
