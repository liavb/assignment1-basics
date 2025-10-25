#!/usr/bin/env python3
"""
Quick demo: Train a tiny transformer and generate text in under 30 seconds.
This is a minimal working example that demonstrates the complete pipeline.

Note: This uses the task 7.2 model architecture but with reduced training.
For full task 7.2 training (327M tokens), see task_7_2_training.py
"""

import os
import sys
import time
import pickle
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.transformer.model_layers import TransformerLM
from cs336_basics.transformer.optimizer import AdamW
from cs336_basics.transformer.data_utils import load_dataset_mmap, get_batch
from cs336_basics.transformer.nn_utils import crossEntropy
from cs336_basics.tokenizer.tokenizer_obj import Tokenizer
from cs336_basics.decoder import decode_with_tokenizer


def load_tokenizer():
    """Load the TinyStories tokenizer"""
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    vocab_path = os.path.join(project_root, "cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_vocab_vocab_size_10000_num_docs_2413403.pkl")
    merges_path = os.path.join(project_root, "cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_merges_vocab_size_10000_num_docs_2413403.pkl")

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(merges_path, 'rb') as f:
        merges = pickle.load(f)

    return Tokenizer(vocab=vocab, merges=merges, special_tokens=['<|endoftext|>'])


def create_tiny_model(vocab_size):
    """Create model according to task 7.2 specifications (but we'll train it quickly)"""
    return TransformerLM(
        vocab_size=vocab_size,
        context_length=256,  # Task 7.2 spec
        num_layers=4,        # Task 7.2 spec
        d_model=512,         # Task 7.2 spec
        num_heads=16,        # Task 7.2 spec
        d_ff=1344,           # Task 7.2 spec (8/3 * d_model, multiple of 64)
        theta=10000.0        # Task 7.2 spec
    )


def quick_train(model, train_data, device="cpu", iterations=150):
    """Train model very quickly (reduced version of task 7.2 training)"""
    import torch

    model.to(device)
    model.train()

    # Use AdamW with task 7.2 hyperparameters
    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,           # Good default from task 7.2 tuning
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.1
    )

    print(f"Quick training for {iterations} iterations...")

    for i in range(iterations):
        # Small batch for speed (task 7.2 uses batch_size=32, context=256)
        x, y = get_batch(train_data, batch_size=4, context_length=64, device=device)

        # Forward pass
        logits = model(x)
        loss = crossEntropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"  Iteration {i:3d} | Loss: {loss.item():.3f}")

    print("Training complete!")
    return model


def demo_generation(model, tokenizer, device="cpu"):
    """Demo text generation with various prompts"""
    print("\nTesting text generation...")

    prompts = [
        "Once upon a time",
        "The little girl",
        "In the forest"
    ]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")

        try:
            # Generate with different temperatures
            for temp in [0.7, 1.0]:
                generated = decode_with_tokenizer(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=20,
                    temperature=temp,
                    top_p=0.9,
                    device=device
                )

                # Show only the new part
                if generated.startswith(prompt):
                    new_text = generated[len(prompt):].strip()
                    print(f"  Temp {temp}: '{prompt}{new_text}'")
                else:
                    print(f"  Temp {temp}: '{generated}'")

        except Exception as e:
            print(f"  Error: {e}")


def main():
    print("🚀 Quick Transformer Demo (Task 7.2 Architecture)")
    print("=" * 40)

    try:
        # 1. Load tokenizer
        print("1. Loading tokenizer...")
        tokenizer = load_tokenizer()
        print(f"   Vocab size: {len(tokenizer.vocab)}")

        # 2. Create tiny model
        print("2. Creating tiny model...")
        model = create_tiny_model(len(tokenizer.vocab))
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {total_params:,}")

        # 3. Load small amount of training data
        print("3. Loading training data...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        train_data_path = os.path.join(project_root, "cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_tokens_u16.bin")
        train_data = load_dataset_mmap(train_data_path, dtype=np.uint16)
        print(f"   Data size: {len(train_data):,} tokens")

        # 4. Quick training (very fast)
        print("4. Training model...")
        start_time = time.time()
        model = quick_train(model, train_data, iterations=50)
        train_time = time.time() - start_time
        print(f"   Training time: {train_time:.1f} seconds")

        # 5. Test generation
        print("5. Testing generation...")
        demo_generation(model, tokenizer)

        print("\n" + "=" * 40)
        print("✅ Demo completed successfully!")
        print("The model learned to generate text!")
        print("\nℹ️  This was a quick demo with limited training.")
        print("For full Task 7.2 training (327M tokens), run:")
        print("  python examples/task_7_2_training.py")

        # Interactive mode
        print("\n🎯 Try your own prompts:")
        while True:
            try:
                user_prompt = input("\nEnter prompt (or 'quit'): ").strip()
                if user_prompt.lower() in ['quit', 'q', 'exit']:
                    break

                if user_prompt:
                    generated = decode_with_tokenizer(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=user_prompt,
                        max_new_tokens=25,
                        temperature=0.8,
                        top_p=0.9
                    )
                    print(f"Generated: {generated}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("Goodbye! 👋")

    except FileNotFoundError as e:
        print(f"❌ Missing file: {e}")
        print("Please ensure tokenizer files are in cs336_basics/tokenizer/")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
