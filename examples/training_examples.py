#!/usr/bin/env python3
"""
Example training script demonstrating how to use the training system.
This shows different ways to start training with various configurations.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_basic_training():
    """Run basic training with minimal configuration"""
    cmd = [
        sys.executable, "-m", "cs336_basics.train",
        "--train_data_path", "./cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_tokens_u16.bin",
        "--val_data_path", "./cs336_basics/tokenizer/TinyStoriesV2_GPT4_valid_tokens_u16.bin",
        "--vocab_size", "10000",
        "--batch_size", "8",
        "--max_iterations", "1000",
        "--log_interval", "50",
        "--eval_interval", "200",
        "--checkpoint_interval", "500",
        "--d_model", "256",
        "--num_layers", "4",
        "--num_heads", "4",
        "--d_ff", "1024",
        "--context_length", "128"
    ]
    
    print("Running basic training...")
    print(" ".join(cmd))
    subprocess.run(cmd)

def run_config_based_training():
    """Run training using a configuration file"""
    cmd = [
        sys.executable, "-m", "cs336_basics.train",
        "--config", "./example_configs/small_model.json",
        "--train_data_path", "./cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_tokens_u16.bin",
        "--val_data_path", "./cs336_basics/tokenizer/TinyStoriesV2_GPT4_valid_tokens_u16.bin",
        "--max_iterations", "2000"  # Override config setting
    ]
    
    print("Running config-based training...")
    print(" ".join(cmd))
    subprocess.run(cmd)

def run_wandb_training():
    """Run training with Weights & Biases logging"""
    cmd = [
        sys.executable, "-m", "cs336_basics.train",
        "--config", "./example_configs/medium_model.json",
        "--train_data_path", "./cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_tokens_u16.bin",
        "--val_data_path", "./cs336_basics/tokenizer/TinyStoriesV2_GPT4_valid_tokens_u16.bin",
        "--use_wandb",
        "--wandb_run_name", "transformer_experiment_1",
        "--max_iterations", "5000"
    ]
    
    print("Running training with W&B logging...")
    print("Note: Make sure you have wandb installed and are logged in")
    print(" ".join(cmd))
    subprocess.run(cmd)

def resume_training():
    """Example of resuming training from a checkpoint"""
    checkpoint_path = "./checkpoints/checkpoint_iter_500.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found. Run basic training first.")
        return
    
    cmd = [
        sys.executable, "-m", "cs336_basics.train",
        "--config", "./example_configs/small_model.json",
        "--train_data_path", "./cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_tokens_u16.bin",
        "--val_data_path", "./cs336_basics/tokenizer/TinyStoriesV2_GPT4_valid_tokens_u16.bin",
        "--resume_from_checkpoint", checkpoint_path,
        "--max_iterations", "2000"
    ]
    
    print("Resuming training from checkpoint...")
    print(" ".join(cmd))
    subprocess.run(cmd)

def hyperparameter_sweep():
    """Example of running multiple experiments with different hyperparameters"""
    base_cmd = [
        sys.executable, "-m", "cs336_basics.train",
        "--train_data_path", "./cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_tokens_u16.bin",
        "--val_data_path", "./cs336_basics/tokenizer/TinyStoriesV2_GPT4_valid_tokens_u16.bin",
        "--vocab_size", "10000",
        "--max_iterations", "1000",
        "--log_interval", "100",
        "--eval_interval", "250"
    ]
    
    # Test different learning rates
    learning_rates = [1e-4, 3e-4, 6e-4]
    
    for lr in learning_rates:
        print(f"\n=== Training with learning rate {lr} ===")
        cmd = base_cmd + [
            "--learning_rate", str(lr),
            "--checkpoint_dir", f"./checkpoints_lr_{lr}",
            "--log_dir", f"./logs_lr_{lr}",
            "--batch_size", "16",
            "--d_model", "256",
            "--num_layers", "4",
            "--num_heads", "4"
        ]
        
        print(" ".join(cmd))
        subprocess.run(cmd)

if __name__ == "__main__":
    print("Transformer Training Examples")
    print("=============================")
    
    if len(sys.argv) < 2:
        print("\nAvailable examples:")
        print("  basic     - Run basic training")
        print("  config    - Run training with config file")
        print("  wandb     - Run training with W&B logging")
        print("  resume    - Resume training from checkpoint")
        print("  sweep     - Run hyperparameter sweep")
        print("\nUsage: python examples/training_examples.py <example_name>")
        sys.exit(1)
    
    example = sys.argv[1].lower()
    
    if example == "basic":
        run_basic_training()
    elif example == "config":
        run_config_based_training()
    elif example == "wandb":
        run_wandb_training()
    elif example == "resume":
        resume_training()
    elif example == "sweep":
        hyperparameter_sweep()
    else:
        print(f"Unknown example: {example}")
        sys.exit(1)
