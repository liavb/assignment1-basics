#!/usr/bin/env python3
"""
Task 7.2: Simple CPU Training for TinyStories
Minimal, fast training script for CPU.
"""

import os
import sys
import time
import pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.transformer.model_layers import TransformerLM
from cs336_basics.transformer.optimizer import AdamW
from cs336_basics.transformer.data_utils import load_dataset_mmap, get_batch
from cs336_basics.transformer.nn_utils import crossEntropy
from cs336_basics.tokenizer.tokenizer_obj import Tokenizer
from cs336_basics.decoder import decode_with_tokenizer

# Performance optimizations for CPU
torch.set_num_threads(os.cpu_count() or 4)  # Use all CPU cores
torch.set_flush_denormal(True)  # Improve numerical performance

# ============================================================
# CONFIGURATION - Edit these to change training
# ============================================================
BATCH_SIZE = 16          # Small for CPU speed
CONTEXT_LENGTH = 128    # Short for CPU speed
TOTAL_STEPS = 350      # Quick training
LEARNING_RATE = 3e-4
EVAL_EVERY = 50        # Evaluate every N steps
PRINT_EVERY = 10        # Print progress every N steps

# Resume training from checkpoint
RESUME_FROM_CHECKPOINT = "models/task_7_2_model_final.pt" # Set to model path to resume, or None to train from scratch
# Example: RESUME_FROM_CHECKPOINT = "models/task_7_2_model_final.pt"


def load_tokenizer(project_root):
    """Load the TinyStories tokenizer"""
    vocab_path = os.path.join(project_root, "cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_vocab_vocab_size_10000_num_docs_2413403.pkl")
    merges_path = os.path.join(project_root, "cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_merges_vocab_size_10000_num_docs_2413403.pkl")

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(merges_path, 'rb') as f:
        merges = pickle.load(f)

    return Tokenizer(vocab=vocab, merges=merges, special_tokens=['<|endoftext|>'])


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Load a checkpoint and resume training.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model instance to load weights into
        optimizer: Optimizer instance to load state into (optional)
        device: Device to load tensors to

    Returns:
        Dictionary with checkpoint info (step, loss, config)
    """
    print(f"\n📂 Loading checkpoint from: {checkpoint_path}", flush=True)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("   ✅ Model weights loaded", flush=True)

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("   ✅ Optimizer state loaded", flush=True)

    # Display checkpoint info
    start_step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', 'unknown')
    config = checkpoint.get('config', {})

    print(f"   📊 Checkpoint info:", flush=True)
    print(f"      Step: {start_step}", flush=True)
    print(f"      Loss: {loss}", flush=True)
    if config:
        print(f"      Config: {config}", flush=True)

    return {
        'step': start_step,
        'loss': loss,
        'config': config
    }


def demo_sentence_completion(model, tokenizer, device="cpu"):
    """Demo text generation with various prompts"""
    print("\n" + "="*60, flush=True)
    print("SENTENCE COMPLETION DEMO", flush=True)
    print("="*60, flush=True)

    prompts = [
        "Once upon a time",
        "The little girl",
        "In the forest",
        "The big red"
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
                    max_new_tokens=30,
                    temperature=temp,
                    top_p=0.9,
                    device=device
                )

                # Show the generated text
                print(f"   🌡️ Temp {temp}: {generated}", flush=True)

        except Exception as e:
            print(f"   ❌ Error: {e}", flush=True)

    print("\n" + "="*60, flush=True)


def main():
    print("="*60, flush=True)
    print("SIMPLE CPU TRAINING - Task 7.2", flush=True)
    print("="*60, flush=True)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Load tokenizer
    print("\n[1/6] Loading tokenizer...", flush=True)
    tokenizer = load_tokenizer(project_root)
    vocab_size = len(tokenizer.vocab)
    print(f"      Vocab size: {vocab_size}", flush=True)

    # Load data
    print("\n[2/6] Loading data...", flush=True)
    train_path = os.path.join(project_root, "cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_tokens_u16.bin")
    valid_path = os.path.join(project_root, "cs336_basics/tokenizer/TinyStoriesV2_GPT4_valid_tokens_u16.bin")

    train_data = load_dataset_mmap(train_path, dtype=np.uint16)
    valid_data = load_dataset_mmap(valid_path, dtype=np.uint16)
    print(f"      Train: {len(train_data):,} tokens", flush=True)
    print(f"      Valid: {len(valid_data):,} tokens", flush=True)

    # Create model
    print("\n[3/6] Creating model...", flush=True)
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        theta=10000.0
    )
    model.to("cpu")
    model.train()

    params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {params:,}", flush=True)

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)

    # Resume from checkpoint if specified
    start_step = 0
    if RESUME_FROM_CHECKPOINT is not None:
        checkpoint_path = os.path.join(project_root, RESUME_FROM_CHECKPOINT) if not os.path.isabs(RESUME_FROM_CHECKPOINT) else RESUME_FROM_CHECKPOINT
        checkpoint_info = load_checkpoint(checkpoint_path, model, optimizer, device="cpu")
        start_step = checkpoint_info['step']
        print(f"\n   🔄 Resuming from step {start_step}", flush=True)
        print(f"   Will train for {TOTAL_STEPS} MORE steps (total: {start_step + TOTAL_STEPS})", flush=True)

    # Training info
    print("\n[4/6] Training setup:", flush=True)
    print(f"      Batch size: {BATCH_SIZE}", flush=True)
    print(f"      Context: {CONTEXT_LENGTH}", flush=True)
    print(f"      Steps: {TOTAL_STEPS} {'(additional)' if start_step > 0 else ''}", flush=True)
    print(f"      Starting from step: {start_step}", flush=True)
    print(f"      LR: {LEARNING_RATE}", flush=True)
    total_tokens = BATCH_SIZE * CONTEXT_LENGTH * TOTAL_STEPS
    print(f"      Total tokens (this session): {total_tokens:,} (~{total_tokens/1e6:.1f}M)", flush=True)

    # Train
    print("\n[5/6] Training...", flush=True)
    print("-"*60, flush=True)

    import torch
    start_time = time.time()

    for step in range(TOTAL_STEPS):
        # Training step
        x, y = get_batch(train_data, batch_size=BATCH_SIZE, context_length=CONTEXT_LENGTH, device="cpu")

        logits = model(x)
        loss = crossEntropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Current absolute step (for checkpointing)
        current_step = start_step + step + 1

        # Print progress
        if step % PRINT_EVERY == 0 or step == TOTAL_STEPS - 1:
            elapsed = time.time() - start_time
            tokens_done = (step + 1) * BATCH_SIZE * CONTEXT_LENGTH
            percent = 100 * (step + 1) / TOTAL_STEPS

            # Calculate ETA
            if step > 0:
                time_per_step = elapsed / (step + 1)
                eta_sec = time_per_step * (TOTAL_STEPS - step - 1)
                eta_str = f"{eta_sec/60:.1f}min"
            else:
                eta_str = "calculating..."

            print(f"Step {step+1:4d}/{TOTAL_STEPS} (abs: {current_step}) [{percent:5.1f}%] | "
                  f"Loss: {loss.item():.4f} | "
                  f"Tokens: {tokens_done:,} | "
                  f"ETA: {eta_str}", flush=True)

        # Evaluate
        if step % EVAL_EVERY == 0 and step > 0:
            model.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for _ in range(20):  # 20 batches
                    x, y = get_batch(valid_data, batch_size=BATCH_SIZE, context_length=CONTEXT_LENGTH, device="cpu")
                    logits = model(x)
                    loss = crossEntropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    eval_loss += loss.item()
            eval_loss /= 20
            model.train()
            print(f"      >>> Validation loss: {eval_loss:.4f}", flush=True)

    # Final evaluation
    print("\n" + "-"*60, flush=True)
    print("FINAL EVALUATION", flush=True)
    print("-"*60, flush=True)

    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for _ in range(50):
            x, y = get_batch(valid_data, batch_size=BATCH_SIZE, context_length=CONTEXT_LENGTH, device="cpu")
            logits = model(x)
            loss = crossEntropy(logits.view(-1, logits.size(-1)), y.view(-1))
            eval_loss += loss.item()
    eval_loss /= 30

    total_time = time.time() - start_time

    print(f"\nTraining completed in {total_time/60:.1f} minutes", flush=True)
    print(f"Final validation loss: {eval_loss:.4f}", flush=True)

    # Save model
    save_dir = os.path.join(project_root, "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "task_7_2_model_final.pt")

    # Get original model if compiled
    model_to_save = model
    if hasattr(model, '_orig_mod'):
        model_to_save = model._orig_mod

    # Calculate final absolute step count
    final_step = start_step + TOTAL_STEPS

    torch.save({
        'step': final_step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': eval_loss,
        'config': {
            'vocab_size': vocab_size,
            'batch_size': BATCH_SIZE,
            'context_length': CONTEXT_LENGTH,
            'learning_rate': LEARNING_RATE,
        }
    }, save_path)

    print(f"\n[SAVED] Model saved to: {save_path}", flush=True)
    print(f"        Final step: {final_step}", flush=True)

    # Demo sentence completion
    print("\n[6/6] Testing model with sentence completion...", flush=True)
    demo_sentence_completion(model_to_save, tokenizer, device="cpu")

    print("\n" + "="*60, flush=True)
    print("TRAINING COMPLETE!", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()

