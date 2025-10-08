#!/usr/bin/env python3
"""
Demonstration of memory-mapped dataset loading and validation.

This script shows how to use the memory mapping functionality to efficiently
load and sample from large tokenized datasets.
"""

import numpy as np
from cs336_basics.transformer.data_utils import load_dataset_mmap, get_batch

def demo_memory_mapping():
    """Demonstrate memory mapping with the TinyStories dataset."""

    # Check if we have tokenized datasets available
    train_tokens_path = "cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_tokens_u16.bin"
    valid_tokens_path = "cs336_basics/tokenizer/TinyStoriesV2_GPT4_valid_tokens_u16.bin"

    try:
        print("Loading training dataset with memory mapping...")
        train_dataset = load_dataset_mmap(train_tokens_path, dtype=np.uint16)
        print(f"✓ Loaded training dataset: shape={train_dataset.shape}, dtype={train_dataset.dtype}")

        # Validate the dataset
        print("\nValidating training dataset...")
        validate_mmap_dataset(train_dataset, vocab_size=10000, sample_size=5000)

        print("\nLoading validation dataset with memory mapping...")
        valid_dataset = load_dataset_mmap(valid_tokens_path, dtype=np.uint16)
        print(f"✓ Loaded validation dataset: shape={valid_dataset.shape}, dtype={valid_dataset.dtype}")

        # Validate the validation dataset
        print("\nValidating validation dataset...")
        validate_mmap_dataset(valid_dataset, vocab_size=10000, sample_size=2000)

        # Sample a batch from the training dataset
        print("\nSampling a batch from the training dataset...")
        batch_size = 4
        context_length = 16
        x, y = get_batch(train_dataset, batch_size, context_length, device='cpu')

        print(f"Input batch shape: {x.shape}")
        print(f"Target batch shape: {y.shape}")
        print(f"Sample input sequence: {x[0].tolist()}")
        print(f"Sample target sequence: {y[0].tolist()}")

        # Verify that targets are shifted by 1
        assert torch.all(x[0, 1:] == y[0, :-1]), "Target should be input shifted by 1"
        print("✓ Target sequences correctly shifted by 1")

        # Sample a batch from the validation dataset
        print("\nSampling a batch from the validation dataset...")
        x_val, y_val = get_batch(valid_dataset, batch_size, context_length, device='cpu')
        print(f"Validation input batch shape: {x_val.shape}")
        print(f"Validation target batch shape: {y_val.shape}")

        print("\n✓ Memory mapping demonstration completed successfully!")

    except FileNotFoundError as e:
        print(f"Dataset files not found: {e}")
        print("This is expected if you haven't tokenized the datasets yet.")

        # Create a small demo with synthetic data
        print("\nCreating synthetic dataset for demonstration...")
        synthetic_data = np.random.randint(0, 1000, size=10000, dtype=np.uint16)

        # Save as binary file
        synthetic_path = "demo_synthetic_tokens.bin"
        synthetic_data.tofile(synthetic_path)

        # Load with memory mapping
        mmap_data = load_dataset_mmap(synthetic_path, dtype=np.uint16)
        validate_mmap_dataset(mmap_data, vocab_size=1000, sample_size=500)

        # Sample a batch
        x, y = get_batch(mmap_data, batch_size=4, context_length=8, device='cpu')
        print(f"Synthetic data batch shape: {x.shape}")
        print(f"Sample sequence: {x[0].tolist()}")

        # Clean up
        import os
        os.remove(synthetic_path)
        print("✓ Synthetic data demonstration completed!")


def validate_mmap_dataset(dataset: np.memmap, vocab_size: int, sample_size: int = 1000) -> None:
    """
    Validate that a memory-mapped dataset contains expected values.

    Args:
        dataset: Memory-mapped dataset array
        vocab_size: Expected vocabulary size (max token ID + 1)
        sample_size: Number of random samples to check
    """
    # Check a random sample of the dataset
    if len(dataset) > sample_size:
        indices = np.random.choice(len(dataset), size=sample_size, replace=False)
        sample_values = dataset[indices]
    else:
        sample_values = dataset[:]

    min_val = np.min(sample_values)
    max_val = np.max(sample_values)

    print(f"Dataset validation:")
    print(f"  Shape: {dataset.shape}")
    print(f"  Dtype: {dataset.dtype}")
    print(f"  Min value: {min_val}")
    print(f"  Max value: {max_val}")
    print(f"  Expected vocab size: {vocab_size}")

    if min_val < 0:
        raise ValueError(f"Dataset contains negative values: min={min_val}")
    if max_val >= vocab_size:
        raise ValueError(f"Dataset contains values >= vocab_size: max={max_val}, vocab_size={vocab_size}")

    print("  ✓ Dataset validation passed")



if __name__ == "__main__":
    import torch
    demo_memory_mapping()


