#!/usr/bin/env python3
"""
Test a saved model from Task 7.2 training.
This script loads the trained model and performs sentence completion.
"""

import os
import sys
import pickle
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.transformer.model_layers import TransformerLM
from cs336_basics.tokenizer.tokenizer_obj import Tokenizer
from cs336_basics.decoder import decode_with_tokenizer


def load_tokenizer(project_root):
    """Load the TinyStories tokenizer"""
    vocab_path = os.path.join(project_root, "cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_vocab_vocab_size_10000_num_docs_2413403.pkl")
    merges_path = os.path.join(project_root, "cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_merges_vocab_size_10000_num_docs_2413403.pkl")

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(merges_path, 'rb') as f:
        merges = pickle.load(f)

    return Tokenizer(vocab=vocab, merges=merges, special_tokens=['<|endoftext|>'])


def load_trained_model(model_path, vocab_size):
    """Load a saved model checkpoint"""
    print(f"Loading model from: {model_path}", flush=True)

    checkpoint = torch.load(model_path, map_location='cpu')

    # Create model with same architecture
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        theta=10000.0
    )

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully!", flush=True)
    print(f"  Training step: {checkpoint.get('step', 'unknown')}", flush=True)
    print(f"  Validation loss: {checkpoint.get('loss', 'unknown'):.4f}", flush=True)

    return model, checkpoint


def interactive_generation(model, tokenizer, device="cpu"):
    """Interactive text generation loop"""
    print("\n" + "="*60)
    print("INTERACTIVE TEXT GENERATION")
    print("="*60)
    print("Enter prompts to generate text. Type 'quit' to exit.")
    print("-"*60)

    while True:
        try:
            prompt = input("\n📝 Enter prompt: ").strip()

            if prompt.lower() in ['quit', 'q', 'exit']:
                print("Goodbye! 👋")
                break

            if not prompt:
                continue

            print(f"\nGenerating (temp=0.8, top_p=0.9)...", flush=True)

            generated = decode_with_tokenizer(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=50,
                temperature=0.8,
                top_p=0.9,
                device=device
            )

            print(f"\n✨ Generated:\n{generated}\n")
            print("-"*60)

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def demo_prompts(model, tokenizer, device="cpu"):
    """Demo text generation with predefined prompts"""
    print("\n" + "="*60)
    print("SENTENCE COMPLETION DEMO")
    print("="*60)

    prompts = [
        "Once upon a time",
        "The little girl",
        "In the forest",
        "The big red",
        "One day a boy"
    ]

    model.eval()

    for prompt in prompts:
        print(f"\n📝 Prompt: '{prompt}'", flush=True)

        try:
            # Generate with different temperatures
            for temp in [0.7, 1.0]:
                generated = decode_with_tokenizer(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=40,
                    temperature=temp,
                    top_p=0.9,
                    device=device
                )

                print(f"   🌡️ Temp {temp}: {generated}", flush=True)

        except Exception as e:
            print(f"   ❌ Error: {e}", flush=True)

    print("\n" + "="*60)


def main():
    print("="*60)
    print("TEST SAVED MODEL - Task 7.2")
    print("="*60)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    model_path = os.path.join(project_root, "models/task_7_2_model_final.pt")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}")
        print("Please train the model first by running:")
        print("  python examples/task_7_2_training.py")
        return

    # Load tokenizer
    print("\n[1/3] Loading tokenizer...", flush=True)
    tokenizer = load_tokenizer(project_root)
    vocab_size = len(tokenizer.vocab)
    print(f"      Vocab size: {vocab_size}", flush=True)

    # Load model
    print("\n[2/3] Loading trained model...", flush=True)
    model, checkpoint = load_trained_model(model_path, vocab_size)

    # Demo generation
    print("\n[3/3] Testing text generation...", flush=True)
    demo_prompts(model, tokenizer, device="cpu")

    # Interactive mode
    print("\n")
    response = input("Would you like to try interactive generation? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        interactive_generation(model, tokenizer, device="cpu")

    print("\n" + "="*60)

