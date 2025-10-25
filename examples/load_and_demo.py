#!/usr/bin/env python3
"""
Load a saved model and apply demo_sentence_completion.
This script loads any saved model checkpoint and runs text generation demos.
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


def demo_sentence_completion(model, tokenizer, device="cpu"):
    """Demo text generation with various prompts"""
    print("\n" + "="*60, flush=True)
    print("SENTENCE COMPLETION DEMO", flush=True)
    print("="*60, flush=True)

    prompts = [
        "Once upon a time",
        "The little girl",
        "In the forest",
        # "The big red",
        # "One day a boy"
    ]

    model.eval()

    for prompt in prompts:
        print(f"\n📝 Prompt: '{prompt}'", flush=True)

        try:
            # Generate with different temperatures
            for temp in [0.1, 0.5, 1]:
                generated = decode_with_tokenizer(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=50,
                    temperature=temp,
                    top_p=0.9,
                    device=device
                )

                # Show the generated text
                print(f"   🌡️ Temp {temp}: {generated}", flush=True)

        except Exception as e:
            print(f"   ❌ Error: {e}", flush=True)

    print("\n" + "="*60, flush=True)


def load_saved_model(model_path, vocab_size, device="cpu"):
    """Load a saved model checkpoint"""
    print(f"Loading model from: {model_path}", flush=True)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Get config from checkpoint if available
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"Model config found in checkpoint:", flush=True)
        for key, value in config.items():
            print(f"  {key}: {value}", flush=True)

    # Create model with Task 7.2 architecture
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=128,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        theta=10000.0
    )

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"\n✅ Model loaded successfully!", flush=True)
    print(f"   Training step: {checkpoint.get('step', 'unknown')}", flush=True)
    print(f"   Validation loss: {checkpoint.get('loss', 'unknown'):.4f}", flush=True)

    return model


def interactive_mode(model, tokenizer, device="cpu"):
    """Interactive text generation"""
    print("\n" + "="*60, flush=True)
    print("INTERACTIVE MODE", flush=True)
    print("="*60, flush=True)
    print("Type your prompts below. Type 'quit' to exit.", flush=True)
    print("-"*60, flush=True)

    while True:
        try:
            prompt = input("\n📝 Enter prompt: ").strip()

            if prompt.lower() in ['quit', 'q', 'exit']:
                print("Goodbye! 👋", flush=True)
                break

            if not prompt:
                continue

            print(f"\nGenerating (temp=0.25, top_p=0.9)...", flush=True)

            generated = decode_with_tokenizer(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=75,
                temperature=0.25,
                top_p=0.9,
                device=device
            )

            print(f"\n✨ Generated:\n{generated}\n", flush=True)
            print("-"*60, flush=True)

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋", flush=True)
            break
        except Exception as e:
            print(f"❌ Error: {e}", flush=True)


def main():
    print("="*60, flush=True)
    print("LOAD MODEL & DEMO SENTENCE COMPLETION", flush=True)
    print("="*60, flush=True)

    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Default model path (you can change this)
    default_model = os.path.join(project_root, "models/task_7_2_model_final.pt")

    # Ask user for model path or use default
    print(f"\nDefault model: {default_model}")
    user_input = input("Enter model path (or press Enter for default): ").strip()

    model_path = user_input if user_input else default_model

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}", flush=True)
        print("\nAvailable models in models/:", flush=True)
        models_dir = os.path.join(project_root, "models")
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.pt'):
                    print(f"  - {f}", flush=True)
        else:
            print("  (models directory not found)", flush=True)
        print("\nPlease train a model first by running:", flush=True)
        print("  python examples/task_7_2_training.py", flush=True)
        return

    # Load tokenizer
    print("\n[1/3] Loading tokenizer...", flush=True)
    tokenizer = load_tokenizer(project_root)
    vocab_size = len(tokenizer.vocab)
    print(f"      Vocab size: {vocab_size}", flush=True)

    # Load model
    print("\n[2/3] Loading model...", flush=True)
    model = load_saved_model(model_path, vocab_size, device="cpu")

    # Run demo
    print("\n[3/3] Running demo...", flush=True)
    demo_sentence_completion(model, tokenizer, device="cpu")

    # Ask if user wants interactive mode
    print("\n", flush=True)
    response = input("Try interactive mode? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        interactive_mode(model, tokenizer, device="cpu")

    print("\n" + "="*60, flush=True)
    print("DONE!", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()

