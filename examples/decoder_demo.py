#!/usr/bin/env python3
"""
Example script demonstrating how to use the text generation decoder.
"""

import sys
import os
import torch
import pickle

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.decoder import (
    generate_text_simple,
    decode_with_tokenizer,
    temperature_scaled_softmax,
    top_p_sampling
)
from cs336_basics.transformer.model_layers import TransformerLM
from cs336_basics.tokenizer.tokenizer_obj import Tokenizer


def demo_temperature_and_top_p():
    """Demonstrate temperature scaling and top-p sampling effects."""
    print("=== Temperature Scaling Demo ===")

    # Create example logits (representing a prediction over 5 tokens)
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.5, 0.1]])

    print("Original logits:", logits.squeeze().tolist())

    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    for temp in temperatures:
        probs = temperature_scaled_softmax(logits, temp)
        print(f"Temperature {temp:3.1f}: {[f'{p:.3f}' for p in probs.squeeze().tolist()]}")

    print("\n=== Top-P Sampling Demo ===")

    # Use probabilities from temperature = 1.0
    probs = temperature_scaled_softmax(logits, 1.0)
    print("Original probs: ", [f'{p:.3f}' for p in probs.squeeze().tolist()])

    p_values = [0.5, 0.7, 0.9, 1.0]
    for p in p_values:
        filtered_probs = top_p_sampling(probs, p)
        non_zero = [(i, f'{prob:.3f}') for i, prob in enumerate(filtered_probs.squeeze().tolist()) if prob > 0]
        print(f"Top-p {p:3.1f}: {non_zero}")


def demo_with_dummy_model():
    """Demonstrate text generation with a small dummy model."""
    print("\n=== Dummy Model Generation Demo ===")

    # Create a small dummy model for demonstration
    vocab_size = 100
    context_length = 64
    d_model = 128

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=2,
        d_model=d_model,
        num_heads=4,
        d_ff=256,
        theta=10000.0
    )

    # Create a dummy prompt (random token IDs)
    prompt_ids = [1, 2, 3, 4, 5]  # Some dummy tokens

    print(f"Input prompt tokens: {prompt_ids}")

    # Generate with different settings
    scenarios = [
        {"name": "Greedy (temp=0.1)", "temperature": 0.1, "top_p": None},
        {"name": "Balanced (temp=1.0)", "temperature": 1.0, "top_p": None},
        {"name": "Creative (temp=1.5)", "temperature": 1.5, "top_p": None},
        {"name": "Nucleus (temp=1.0, p=0.9)", "temperature": 1.0, "top_p": 0.9},
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        generated = generate_text_simple(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=10,
            temperature=scenario['temperature'],
            top_p=scenario['top_p'],
            device="cpu"
        )
        new_tokens = generated[len(prompt_ids):]
        print(f"Generated tokens: {new_tokens}")


def demo_with_real_tokenizer():
    """Demonstrate text generation with a real tokenizer (if available)."""
    print("\n=== Real Tokenizer Demo ===")

    # Try to load a real tokenizer
    tokenizer_paths = [
        "../cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_vocab_vocab_size_10000_num_docs_2413403.pkl",
        # "../open_web_vocab_vocab_size_1000_num_docs_6458.pkl"
    ]

    merges_paths = [
        "../cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_merges_vocab_size_10000_num_docs_2413403.pkl",
        # "../open_web_merges_vocab_size_1000_num_docs_6458.pkl"
    ]

    tokenizer = None
    for vocab_path, merges_path in zip(tokenizer_paths, merges_paths):
        if os.path.exists(vocab_path) and os.path.exists(merges_path):
            try:
                with open(vocab_path, 'rb') as f:
                    vocab = pickle.load(f)
                with open(merges_path, 'rb') as f:
                    merges = pickle.load(f)

                tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=['<|endoftext|>'])
                print(f"Loaded tokenizer with vocab size: {len(vocab)}")
                break
            except Exception as e:
                print(f"Failed to load tokenizer from {vocab_path}: {e}")
                continue

    if tokenizer is None:
        print("No tokenizer found. Skipping real tokenizer demo.")
        return

    # Create a dummy model with the right vocab size
    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=128,
        num_layers=2,
        d_model=256,
        num_heads=8,
        d_ff=512,
        theta=10000.0
    )

    # Test with a simple prompt
    prompts = [
        "Once upon a time",
        "The cat sat on",
        "In a land far away"
    ]

    for prompt in prompts:
        try:
            print(f"\nPrompt: '{prompt}'")

            # Encode and show tokens
            prompt_tokens = tokenizer.encode(prompt)
            print(f"Prompt tokens: {prompt_tokens}")

            # Generate with different temperatures
            for temp in [0.1, 0.5, 1.0]:
                generated_text = decode_with_tokenizer(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=10,
                    temperature=temp,
                    top_p=0.9,
                    device="cpu"
                )
                print(f"Temperature {temp}: '{generated_text}'")

        except Exception as e:
            print(f"Error generating for prompt '{prompt}': {e}")


def benchmark_generation_speed():
    """Benchmark generation speed."""
    print("\n=== Generation Speed Benchmark ===")

    model = TransformerLM(
        vocab_size=1000,
        context_length=512,
        num_layers=4,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        theta=10000.0
    )

    prompt_ids = [1, 2, 3, 4, 5]

    import time

    # Benchmark different sequence lengths
    for max_tokens in [10, 50, 100]:
        start_time = time.time()

        generated = generate_text_simple(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_tokens,
            temperature=1.0,
            device="cpu"
        )

        end_time = time.time()
        tokens_per_sec = max_tokens / (end_time - start_time)

        print(f"Generated {max_tokens} tokens in {end_time - start_time:.2f}s ({tokens_per_sec:.1f} tokens/sec)")


if __name__ == "__main__":
    print("Text Generation Decoder Demo")
    print("=" * 50)

    # Run all demos
    # demo_temperature_and_top_p()
    # demo_with_dummy_model()
    demo_with_real_tokenizer()
    # benchmark_generation_speed()

    print("\nDemo complete!")
