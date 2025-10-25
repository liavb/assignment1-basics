import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch

# Make TensorBoard import optional
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from cs336_basics.transformer.model_layers import TransformerLM
from cs336_basics.transformer.optimizer import AdamW
from cs336_basics.transformer.data_utils import load_dataset_mmap, get_batch, save_checkpoint, load_checkpoint
from cs336_basics.transformer.nn_utils import get_lr_cosine_schedule, gradient_clipping, crossEntropy


class TrainingConfig:
    """Configuration class for training hyperparameters"""

    def __init__(self):
        # Model hyperparameters
        self.vocab_size: int = 10000
        self.context_length: int = 512
        self.d_model: int = 768
        self.num_layers: int = 12
        self.num_heads: int = 12
        self.d_ff: int = 3072
        self.rope_theta: float = 10000.0

        # Training hyperparameters
        self.batch_size: int = 32
        self.max_iterations: int = 100000
        self.learning_rate: float = 3e-4
        self.min_learning_rate: float = 3e-5
        self.warmup_iterations: int = 2000
        self.weight_decay: float = 0.1
        self.beta1: float = 0.9
        self.beta2: float = 0.95
        self.grad_clip: float = 1.0

        # Data paths
        self.train_data_path: str = ""
        self.val_data_path: str = ""
        self.checkpoint_dir: str = "./checkpoints"
        self.log_dir: str = "./logs"

        # Logging and evaluation
        self.log_interval: int = 100
        self.eval_interval: int = 1000
        self.eval_iterations: int = 100
        self.checkpoint_interval: int = 5000

        # Resume training
        self.resume_from_checkpoint: Optional[str] = None

        # Device
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # Data type
        self.dtype: str = "float32"
        self.data_dtype: str = "uint16"  # For token data

        # Weights & Biases logging
        self.use_wandb: bool = False
        self.wandb_project: str = "transformer-training"
        self.wandb_run_name: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    @classmethod
    def from_json(cls, json_path: str) -> 'TrainingConfig':
        """Load config from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save_json(self, json_path: str):
        """Save config to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class Trainer:
    """Main training class"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Set up directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Initialize logging
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=config.log_dir)
        else:
            self.writer = None

        # Initialize Weights & Biases if requested
        if config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=config.to_dict()
                )
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed, falling back to tensorboard only")
                self.wandb = None
        else:
            self.wandb = None

        # Load datasets
        print(f"Loading training data from {config.train_data_path}")
        self.train_data = load_dataset_mmap(config.train_data_path, dtype=getattr(np, config.data_dtype))
        print(f"Training data size: {len(self.train_data):,} tokens")

        if config.val_data_path:
            print(f"Loading validation data from {config.val_data_path}")
            self.val_data = load_dataset_mmap(config.val_data_path, dtype=getattr(np, config.data_dtype))
            print(f"Validation data size: {len(self.val_data):,} tokens")
        else:
            self.val_data = None

        # Initialize model
        self.model = TransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            theta=config.rope_theta
        ).to(self.device)

        # Apply torch.compile for speedup based on device
        device_type = str(self.device.type)
        if device_type == "cpu":
            print("Compiling model for CPU with torch.compile...", flush=True)
            self.model = torch.compile(self.model)
        elif device_type == "mps":
            print("Compiling model for MPS with backend='aot_eager'...", flush=True)
            self.model = torch.compile(self.model, backend="aot_eager")
            print("Note: Not using TF32 kernels on MPS (causes instability)", flush=True)
        elif device_type == "cuda":
            print("Enabling TF32 for CUDA (faster matmul)...", flush=True)
            torch.set_float32_matmul_precision('high')

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Training state
        self.iteration = 0
        self.best_val_loss = float('inf')

        # Resume from checkpoint if specified
        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)

        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Training on device: {self.device}")

    def get_learning_rate(self, iteration: int) -> float:
        """Get learning rate for current iteration using cosine schedule with warmup"""
        return get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=self.config.learning_rate,
            min_learning_rate=self.config.min_learning_rate,
            warmup_iters=self.config.warmup_iterations,
            cosine_cycle_iters=self.config.max_iterations
        )

    def train_step(self) -> Dict[str, float]:
        """Execute one training step"""
        self.model.train()

        # Get batch
        x, y = get_batch(
            self.train_data,
            self.config.batch_size,
            self.config.context_length,
            device=str(self.device)
        )

        # Forward pass
        logits = self.model(x)  # Shape: (batch_size, seq_len, vocab_size)

        # Reshape for loss computation
        logits_flat = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
        targets_flat = y.view(-1)  # (batch_size * seq_len,)

        # Compute loss
        loss = crossEntropy(logits_flat, targets_flat)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        gradient_clipping(self.model.parameters(), self.config.grad_clip)

        # Update learning rate
        lr = self.get_learning_rate(self.iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Optimizer step
        self.optimizer.step()

        return {'loss': loss.item(), 'lr': lr}

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        if self.val_data is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for _ in range(self.config.eval_iterations):
                x, y = get_batch(
                    self.val_data,
                    self.config.batch_size,
                    self.config.context_length,
                    device=str(self.device)
                )

                logits = self.model(x)
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = y.view(-1)

                loss = crossEntropy(logits_flat, targets_flat)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}

    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to tensorboard and wandb"""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.iteration)

        if self.wandb:
            self.wandb.log(metrics, step=self.iteration)

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_iter_{self.iteration}.pt"
        )

        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            iteration=self.iteration,
            out=checkpoint_path
        )

        # Save best model separately
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                iteration=self.iteration,
                out=best_path
            )

        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        self.iteration = load_checkpoint(
            src=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer
        )
        print(f"Resumed training from iteration {self.iteration}")

    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Training for {self.config.max_iterations:,} iterations")

        start_time = time.time()

        while self.iteration < self.config.max_iterations:
            # Training step
            train_metrics = self.train_step()

            # Logging
            if self.iteration % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (self.iteration * self.config.batch_size * self.config.context_length) / elapsed

                print(f"Iteration {self.iteration:6d} | "
                      f"Loss: {train_metrics['loss']:.4f} | "
                      f"LR: {train_metrics['lr']:.2e} | "
                      f"Tokens/sec: {tokens_per_sec:.0f}")

                self.log_metrics(train_metrics)

            # Evaluation
            if self.iteration % self.config.eval_interval == 0 and self.iteration > 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    print(f"Validation | Loss: {eval_metrics['val_loss']:.4f}")
                    self.log_metrics(eval_metrics)

                    # Check if this is the best model
                    if eval_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = eval_metrics['val_loss']
                        self.save_checkpoint(is_best=True)

            # Checkpointing
            if self.iteration % self.config.checkpoint_interval == 0 and self.iteration > 0:
                self.save_checkpoint()

            self.iteration += 1

        # Final checkpoint
        self.save_checkpoint()
        print("Training completed!")

        if self.wandb:
            self.wandb.finish()


def create_config_from_args(args) -> TrainingConfig:
    """Create training config from command line arguments"""
    config = TrainingConfig()

    # Update config with provided arguments
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)

    return config


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer language model")

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=512, help="Context length")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feed-forward dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_iterations", type=int, default=100000, help="Maximum training iterations")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--min_learning_rate", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_iterations", type=int, default=2000, help="Warmup iterations")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")

    # Data paths
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data_path", type=str, help="Path to validation data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Log directory")

    # Logging and evaluation
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Evaluation interval")
    parser.add_argument("--eval_iterations", type=int, default=100, help="Number of evaluation iterations")
    parser.add_argument("--checkpoint_interval", type=int, default=5000, help="Checkpoint interval")

    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")

    # Device and data type
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--data_dtype", type=str, default="uint16", help="Data type for tokens")

    # Weights & Biases
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="transformer-training", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, help="W&B run name")

    # Config file
    parser.add_argument("--config", type=str, help="Path to JSON config file")

    args = parser.parse_args()

    # Load config from file if provided, otherwise create from args
    if args.config:
        config = TrainingConfig.from_json(args.config)
        # Override with any command line arguments
        for key, value in vars(args).items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
    else:
        config = create_config_from_args(args)

    # Set device
    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Save config for reference
    config.save_json(os.path.join(config.checkpoint_dir, "config.json"))

    # Initialize trainer and start training
    trainer = Trainer(config)
    trainer.train()
#
#
# if __name__ == "__main__":
#     main()
