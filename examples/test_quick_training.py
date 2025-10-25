#!/usr/bin/env python3
"""
Quick test script to verify training progress is displayed correctly.
Runs only 50 steps with reduced batch/context for fast CPU testing.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from cs336_basics.transformer.model_layers import TransformerLM
from cs336_basics.transformer.optimizer import AdamW
from cs336_basics.transformer.data_utils import load_dataset_mmap, get_batch
from cs336_basics.transformer.nn_utils import crossEntropy
from cs336_basics.tokenizer.tokenizer_obj import Tokenizer

print("="*60, flush=True)
print("QUICK TRAINING TEST (50 steps, CPU-optimized)", flush=True)
print("="*60, flush=True)

# Load tokenizer
print("\nLoading tokenizer...", flush=True)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

vocab_path = os.path.join(project_root, "cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_vocab_vocab_size_10000_num_docs_2413403.pkl")
merges_path = os.path.join(project_root, "cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_merges_vocab_size_10000_num_docs_2413403.pkl")

with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
with open(merges_path, 'rb') as f:
    merges = pickle.load(f)

tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=['<|endoftext|>'])
print(f"[OK] Vocab size: {len(tokenizer.vocab)}", flush=True)

# Load data
print("\nLoading data...", flush=True)
train_data_path = os.path.join(project_root, "cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_tokens_u16.bin")
train_data = load_dataset_mmap(train_data_path, dtype=np.uint16)
print(f"[OK] Training data: {len(train_data):,} tokens", flush=True)

# Create model
print("\nCreating model...", flush=True)
model = TransformerLM(
    vocab_size=len(tokenizer.vocab),
    context_length=256,
    num_layers=4,
    d_model=512,
    num_heads=16,
    d_ff=1344,
    theta=10000.0
)
model.to("cpu")
model.train()
total_params = sum(p.numel() for p in model.parameters())
print(f"[OK] Model parameters: {total_params:,}", flush=True)

# Create optimizer
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

# Train for 50 steps with smaller batch/context for CPU
print("\n" + "="*60, flush=True)
print("TRAINING (50 steps)", flush=True)
print("="*60, flush=True)

import time
start_time = time.time()
total_steps = 50
batch_size = 16  # Smaller batch for CPU
context_length = 128  # Shorter context for speed

# Calculate tokens processed
tokens_per_step = batch_size * context_length
total_tokens = total_steps * tokens_per_step
print(f"Total tokens to process: {total_tokens:,} (~{total_tokens/1e6:.1f}M)\n", flush=True)

for step in range(total_steps):
    # Get batch
    x, y = get_batch(train_data, batch_size=batch_size, context_length=context_length, device="cpu")

    # Forward pass
    logits = model(x)
    loss = crossEntropy(logits.view(-1, logits.size(-1)), y.view(-1))

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Progress every 5 steps for more frequent updates
    if step % 5 == 0 or step == total_steps - 1:
        elapsed = time.time() - start_time
        percent = 100 * (step + 1) / total_steps
        tokens_processed = (step + 1) * tokens_per_step
        eta_seconds = (elapsed / (step + 1)) * (total_steps - step - 1) if step > 0 else 0
        eta_minutes = eta_seconds / 60

        print(f"[{percent:5.1f}%] Step {step+1:3d}/{total_steps} | "
              f"Loss: {loss.item():.4f} | "
              f"Tokens: {tokens_processed:,} | "
              f"ETA: {eta_minutes:.1f}min", flush=True)

total_time = time.time() - start_time
print(f"\n[OK] Completed in {total_time:.1f} seconds", flush=True)
print(f"[OK] Average: {total_time/total_steps:.2f} sec/step", flush=True)

# Test model saving
print("\nTesting model save...", flush=True)
import torch
save_path = os.path.join(project_root, "models/test_checkpoint.pt")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save({
    'step': total_steps,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}, save_path)
print(f"[OK] Model saved to: {save_path}", flush=True)

print("\n" + "="*60, flush=True)
print("TEST COMPLETED SUCCESSFULLY!", flush=True)
print("="*60, flush=True)

