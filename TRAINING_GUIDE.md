# Training Script Usage Guide

## Overview

The training script (`cs336_basics/train.py`) provides a complete training system for your Transformer language model with all the requested features:

- ✅ **Configurable hyperparameters** via command line or JSON config files
- ✅ **Memory-efficient data loading** using `np.memmap` for large datasets
- ✅ **Robust checkpointing** with automatic saving and resumption
- ✅ **Comprehensive logging** with TensorBoard and optional Weights & Biases

## Quick Start

### Basic Training Command
```bash
python -m cs336_basics.train \
  --train_data_path ./cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_tokens_u16.bin \
  --val_data_path ./cs336_basics/tokenizer/TinyStoriesV2_GPT4_valid_tokens_u16.bin \
  --vocab_size 10000 \
  --batch_size 16 \
  --max_iterations 10000
```

### Using Configuration Files
```bash
python -m cs336_basics.train \
  --config example_configs/small_model.json \
  --train_data_path ./cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_tokens_u16.bin \
  --val_data_path ./cs336_basics/tokenizer/TinyStoriesV2_GPT4_valid_tokens_u16.bin
```

## Key Features

### 1. Flexible Configuration System

**Command Line Arguments:**
- All hyperparameters can be set via command line
- Use `python -m cs336_basics.train --help` to see all options

**JSON Configuration Files:**
- Store configurations in JSON files for reproducibility
- Command line arguments override config file settings
- Example configs provided in `example_configs/`

### 2. Memory-Efficient Data Loading

- Uses `np.memmap` for lazy loading of large datasets
- Only loads required data slices into memory
- Supports both `.npy` and binary files
- Efficient batch sampling from massive datasets

### 3. Comprehensive Checkpointing

**Automatic Saving:**
- Regular checkpoints at configurable intervals
- Best model tracking based on validation loss
- Final checkpoint saved at training completion

**Easy Resumption:**
```bash
python -m cs336_basics.train \
  --resume_from_checkpoint ./checkpoints/checkpoint_iter_5000.pt \
  --config example_configs/small_model.json \
  --train_data_path ./data/train_tokens.bin
```

### 4. Rich Logging and Monitoring

**TensorBoard Integration:**
- Automatic logging to `./logs/` directory
- Training metrics: loss, learning rate, tokens/second
- Validation metrics tracked separately

**Weights & Biases Support:**
```bash
python -m cs336_basics.train \
  --use_wandb \
  --wandb_project "my-transformer-experiments" \
  --wandb_run_name "experiment_1" \
  --config example_configs/medium_model.json \
  --train_data_path ./data/train_tokens.bin
```

## Configuration Options

### Model Hyperparameters
- `vocab_size`: Vocabulary size (default: 10000)
- `context_length`: Maximum sequence length (default: 512)
- `d_model`: Model dimension (default: 768)
- `num_layers`: Number of transformer layers (default: 12)
- `num_heads`: Number of attention heads (default: 12)
- `d_ff`: Feed-forward dimension (default: 3072)
- `rope_theta`: RoPE theta parameter (default: 10000.0)

### Training Hyperparameters
- `batch_size`: Training batch size (default: 32)
- `max_iterations`: Maximum training iterations (default: 100000)
- `learning_rate`: Peak learning rate (default: 3e-4)
- `min_learning_rate`: Minimum learning rate (default: 3e-5)
- `warmup_iterations`: Linear warmup iterations (default: 2000)
- `weight_decay`: AdamW weight decay (default: 0.1)
- `grad_clip`: Gradient clipping norm (default: 1.0)

### Logging and Evaluation
- `log_interval`: Training log frequency (default: 100)
- `eval_interval`: Validation frequency (default: 1000)
- `eval_iterations`: Validation batch count (default: 100)
- `checkpoint_interval`: Checkpoint save frequency (default: 5000)

## Example Workflows

### 1. Quick Experimentation
Use the small model config for fast iterations:
```bash
python -m cs336_basics.train \
  --config example_configs/small_model.json \
  --train_data_path ./data/train_tokens.bin \
  --max_iterations 5000
```

### 2. Hyperparameter Sweep
Create multiple configs and run experiments:
```bash
# Experiment 1: High learning rate
python -m cs336_basics.train \
  --config example_configs/small_model.json \
  --learning_rate 6e-4 \
  --checkpoint_dir ./exp1_checkpoints \
  --log_dir ./exp1_logs \
  --train_data_path ./data/train_tokens.bin

# Experiment 2: Larger model
python -m cs336_basics.train \
  --config example_configs/medium_model.json \
  --checkpoint_dir ./exp2_checkpoints \
  --log_dir ./exp2_logs \
  --train_data_path ./data/train_tokens.bin
```

### 3. Long Training with Monitoring
```bash
python -m cs336_basics.train \
  --config example_configs/medium_model.json \
  --use_wandb \
  --wandb_project "transformer-scaling" \
  --wandb_run_name "768d_12l_baseline" \
  --train_data_path ./data/large_train_tokens.bin \
  --val_data_path ./data/large_val_tokens.bin \
  --max_iterations 100000
```

## Monitoring Training

### View TensorBoard Logs
```bash
tensorboard --logdir ./logs
```

### Check Training Progress
The script outputs regular training updates:
```
Iteration   1000 | Loss: 3.2145 | LR: 2.95e-04 | Tokens/sec: 8192
Validation | Loss: 3.1892
Checkpoint saved to ./checkpoints/checkpoint_iter_1000.pt
```

## Tips for Effective Training

1. **Start Small**: Use `small_model.json` for initial experiments
2. **Monitor Validation**: Watch for overfitting via validation loss
3. **Save Configs**: Always save your configuration for reproducibility
4. **Use Checkpoints**: Resume from checkpoints for long training runs
5. **Log Everything**: Use TensorBoard or W&B to track experiments

## Troubleshooting

**Memory Issues:**
- Reduce `batch_size` or `context_length`
- Use smaller model dimensions

**Slow Training:**
- Increase `batch_size` if memory allows
- Use GPU with `--device cuda`
- Reduce logging frequency

**Data Loading Errors:**
- Verify data paths exist
- Check data format matches `data_dtype` setting
- Ensure sufficient disk space for checkpoints
