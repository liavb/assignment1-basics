#!/usr/bin/env python3
"""
End-to-end example: Train a simple transformer model and generate text.
This script demonstrates the complete pipeline from training to text generation.
"""

import os
import sys
import time
import pickle
import torch
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.transformer.model_layers import TransformerLM
from cs336_basics.transformer.optimizer import AdamW
from cs336_basics.transformer.data_utils import load_dataset_mmap, get_batch, save_checkpoint
from cs336_basics.transformer.nn_utils import get_lr_cosine_schedule, gradient_clipping, crossEntropy
from cs336_basics.tokenizer.tokenizer_obj import Tokenizer
from cs336_basics.decoder import decode_with_tokenizer, generate_text_simple


class SimpleTrainer:
    """Simple trainer for quick experimentation"""

    def __init__(self, model, tokenizer, train_data_path, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)

        # Load training data
        print(f"Loading training data from {train_data_path}")
        self.train_data = load_dataset_mmap(train_data_path, dtype=np.uint16)
        print(f"Training data size: {len(self.train_data):,} tokens")

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )

        self.iteration = 0

    def train_step(self, batch_size=16, context_length=64):
        """Execute one training step"""
        self.model.train()

        # Get batch
        x, y = get_batch(
            self.train_data,
            batch_size,
            context_length,
            device=str(self.device)
        )

        # Forward pass
        logits = self.model(x)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = y.view(-1)

        # Compute loss
        loss = crossEntropy(logits_flat, targets_flat)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        gradient_clipping(self.model.parameters(), 1.0)

        # Learning rate schedule
        lr = get_lr_cosine_schedule(
            it=self.iteration,
            max_learning_rate=3e-4,
            min_learning_rate=3e-5,
            warmup_iters=100,
            cosine_cycle_iters=1000
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Optimizer step
        self.optimizer.step()
        self.iteration += 1

        return loss.item(), lr

    def train(self, num_iterations=1000, batch_size=16, context_length=64, log_interval=50):
        """Train the model for specified iterations"""
        print(f"Starting training for {num_iterations} iterations...")
        print(f"Batch size: {batch_size}, Context length: {context_length}")

        start_time = time.time()

        for i in range(num_iterations):
            loss, lr = self.train_step(batch_size, context_length)

            if i % log_interval == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (i * batch_size * context_length) / elapsed if elapsed > 0 else 0

                print(f"Iteration {i:4d} | Loss: {loss:.4f} | LR: {lr:.2e} | Tokens/sec: {tokens_per_sec:.0f}")

        print("Training completed!")

    def generate_text(self, prompt, max_new_tokens=50, temperature=0.8, top_p=0.9):
        """Generate text from a prompt"""
        self.model.eval()

        return decode_with_tokenizer(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=str(self.device),
            verbose=True
        )


def load_tokenizer():
    """Load the available tokenizer"""
    # Try to load the TinyStories tokenizer
    tokenizer_paths = [
        "./cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_vocab_vocab_size_10000_num_docs_2413403.pkl",
        "./cs336_basics/tokenizer/TinyStoriesV2_GPT4_valid_vocab_vocab_size_10000_num_docs_27630.pkl"
    ]

    merges_paths = [
        "./cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_merges_vocab_size_10000_num_docs_2413403.pkl",
        "./cs336_basics/tokenizer/TinyStoriesV2_GPT4_valid_merges_vocab_size_10000_num_docs_27630.pkl"
    ]

    for vocab_path, merges_path in zip(tokenizer_paths, merges_paths):
        if os.path.exists(vocab_path) and os.path.exists(merges_path):
            try:
                print(f"Loading tokenizer from {vocab_path}")
                with open(vocab_path, 'rb') as f:
                    vocab = pickle.load(f)
                with open(merges_path, 'rb') as f:
                    merges = pickle.load(f)

                tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=['<|endoftext|>'])
                print(f"Loaded tokenizer with vocab size: {len(vocab)}")
                return tokenizer
            except Exception as e:
                print(f"Failed to load tokenizer from {vocab_path}: {e}")
                continue

    raise FileNotFoundError("Could not find tokenizer files. Please ensure TinyStories tokenizer files are available.")


def create_simple_model(vocab_size, device="cpu"):
    """Create a small transformer model for quick training"""
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=256,  # Small context for fast training
        num_layers=6,        # Few layers for fast training
        d_model=256,         # Small model dimension
        num_heads=8,         # Multiple heads but small
        d_ff=1024,           # Small feed-forward dimension
        theta=10000.0
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Created model with {total_params:,} parameters")

    return model


def main():
    print("=" * 60)
    print("End-to-End Transformer Training and Generation Demo")
    print("=" * 60)

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Load tokenizer
        print("\n1. Loading tokenizer...")
        tokenizer = load_tokenizer()

        # Create model
        print("\n2. Creating model...")
        model = create_simple_model(vocab_size=len(tokenizer.vocab), device=device)

        # Find training data
        print("\n3. Finding training data...")
        train_data_paths = [
            "./cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_tokens_u16.bin",
            "./cs336_basics/tokenizer/TinyStoriesV2_GPT4_valid_tokens_u16.bin"
        ]

        train_data_path = None
        for path in train_data_paths:
            if os.path.exists(path):
                train_data_path = path
                break

        if train_data_path is None:
            raise FileNotFoundError("Could not find training data. Please ensure tokenized data files are available.")

        # Initialize trainer
        print("\n4. Initializing trainer...")
        trainer = SimpleTrainer(model, tokenizer, train_data_path, device=device)

        # Train the model
        print("\n5. Training model...")
        trainer.train(
            num_iterations=500,  # Quick training for demo
            batch_size=8,        # Small batch for fast training
            context_length=64,   # Short context for fast training
            log_interval=50
        )

        # Save the trained model
        print("\n6. Saving model...")
        os.makedirs("./models", exist_ok=True)
        save_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            iteration=trainer.iteration,
            out="./models/simple_trained_model.pt"
        )
        print("Model saved to ./models/simple_trained_model.pt")

        # Test text generation
        print("\n7. Testing text generation...")
        test_prompts = [
            "Once upon a time",
            "The little girl",
            "In a magical forest",
            "The cat and the dog",
            "One sunny day"
        ]

        print("\nGenerating text with different settings...")

        for i, prompt in enumerate(test_prompts[:3]):  # Test first 3 prompts
            print(f"\n--- Prompt {i+1}: '{prompt}' ---")

            # Generate with different temperatures
            for temp in [0.1, 0.5, 0.8, 1.2]:
                print(f"\nTemperature {temp}:")
                try:
                    generated = trainer.generate_text(
                        prompt=prompt,
                        max_new_tokens=30,
                        temperature=temp,
                        top_p=0.9
                    )
                    # Extract only the new part (after the prompt)
                    if generated.startswith(prompt):
                        new_part = generated[len(prompt):].strip()
                        print(f"  '{prompt}{new_part}'")
                    else:
                        print(f"  '{generated}'")
                except Exception as e:
                    print(f"  Error: {e}")

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("The model has been trained and can generate text.")
        print("You can now use the saved model for further experiments.")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Please check that all required files are present:")
        print("- Tokenizer files in cs336_basics/tokenizer/")
        print("- Training data files in cs336_basics/tokenizer/")
        return 1

    return 0


def interactive_demo():
    """Interactive demo where user can input prompts"""
    print("\n" + "=" * 40)
    print("Interactive Text Generation")
    print("=" * 40)

    try:
        # Load the trained model
        if not os.path.exists("./models/simple_trained_model.pt"):
            print("No trained model found. Please run the main demo first.")
            return

        # Load tokenizer
        tokenizer = load_tokenizer()

        # Create and load model
        model = create_simple_model(vocab_size=len(tokenizer.vocab))

        checkpoint = torch.load("./models/simple_trained_model.pt", map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained model.")

        trainer = SimpleTrainer(model, tokenizer, "./cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_tokens_u16.bin")

        print("\nEnter prompts to generate text (type 'quit' to exit):")

        while True:
            prompt = input("\nPrompt: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break

            if not prompt:
                continue

            try:
                generated = trainer.generate_text(
                    prompt=prompt,
                    max_new_tokens=40,
                    temperature=0.8,
                    top_p=0.9
                )
                print(f"Generated: {generated}")
            except Exception as e:
                print(f"Error: {e}")

        print("Goodbye!")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="End-to-end transformer training and generation demo")
    parser.add_argument("--interactive", action="store_true", help="Run interactive demo with trained model")
    args = parser.parse_args()

    if args.interactive:
        interactive_demo()
    else:
        main()
