#!/usr/bin/env python3
"""
Quick test to verify the training script works with a minimal example.
"""

import os
import sys
import tempfile
import numpy as np
import torch

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.train import TrainingConfig, Trainer

def create_dummy_data(vocab_size=1000, sequence_length=10000):
    """Create a small dummy dataset for testing"""
    # Generate random token IDs
    data = np.random.randint(0, vocab_size, size=sequence_length, dtype=np.uint16)
    return data

def test_training_script():
    """Test that the training script can run end-to-end"""
    print("Testing training script functionality...")

    # Create temporary files for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy training data
        train_data = create_dummy_data(vocab_size=1000, sequence_length=5000)
        val_data = create_dummy_data(vocab_size=1000, sequence_length=1000)

        train_path = os.path.join(temp_dir, "train_data.bin")
        val_path = os.path.join(temp_dir, "val_data.bin")

        # Save as memory-mapped arrays
        train_data.tofile(train_path)
        val_data.tofile(val_path)

        # Create a minimal training config
        config = TrainingConfig()
        config.vocab_size = 1000
        config.context_length = 64
        config.d_model = 128
        config.num_layers = 2
        config.num_heads = 4
        config.d_ff = 256
        config.batch_size = 4
        config.max_iterations = 10  # Very short for testing
        config.log_interval = 5
        config.eval_interval = 5
        config.checkpoint_interval = 10
        config.train_data_path = train_path
        config.val_data_path = val_path
        config.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        config.log_dir = os.path.join(temp_dir, "logs")
        config.device = "cpu"  # Use CPU for testing

        # Initialize and run trainer
        print("Initializing trainer...")
        trainer = Trainer(config)

        print("Running training loop...")
        trainer.train()

        # Check that checkpoint was created
        checkpoint_files = os.listdir(config.checkpoint_dir)
        assert len(checkpoint_files) > 0, "No checkpoints were created"

        print("✅ Training script test passed!")
        print(f"Created checkpoints: {checkpoint_files}")

if __name__ == "__main__":
    test_training_script()
