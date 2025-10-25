# Task 7.2 Training Guide

This directory contains scripts for training a Transformer model on TinyStories according to the task 7.2 specifications.

## Files

### `quick_demo.py`
A quick demonstration that trains a small model in under a minute to verify your setup works.
- Uses task 7.2 model architecture (17M parameters)
- Trains with minimal iterations for quick verification
- Good for testing before running full training

**Usage:**
```bash
python examples/quick_demo.py
```

### `task_7_2_training.py`
Full task 7.2 training script with learning rate hyperparameter sweep.
- Trains models with different learning rates
- Processes ~327M tokens as required
- Evaluates on validation set
- Saves checkpoints and learning curves
- Target: validation loss ≤ 1.45

**Usage:**
```bash
python examples/task_7_2_training.py
```

## Task 7.2 Specifications

### Model Architecture
- **vocab_size**: 10,000
- **context_length**: 256
- **d_model**: 512
- **d_ff**: 1,344 (≈ 8/3 × d_model, multiple of 64)
- **num_layers**: 4
- **num_heads**: 16
- **RoPE theta**: 10,000
- **Total parameters**: ~17M (non-embedding)

### Training Configuration

**For GPU (CUDA):**
- **Total tokens**: ~327,680,000
- **Target validation loss**: ≤ 1.45
- **Batch size**: 32
- **Context length**: 256
- **Steps**: ~40,000
- **Expected time**: 30-40 minutes on H100

**For CPU/MPS:**
- **Total tokens**: ~40,960,000 (32 × 5000 × 256)
- **Target validation loss**: ≤ 2.00
- **Batch size**: 32
- **Context length**: 256
- **Steps**: 5,000
- **Expected time**: 
  - CPU: ~1 hour 22 minutes (tested on M3 Max, 36GB RAM)
  - MPS: ~36 minutes (tested on M3 Max)

### Device-Specific Optimizations

The training script automatically applies optimizations based on your device:

**CPU:**
- Uses `torch.compile(model)` for JIT compilation speedup
- Reduced token count to 40M for reasonable training time

**MPS (Apple Silicon):**
- Uses `torch.compile(model, backend="aot_eager")` for backward pass optimization
- Does NOT use TF32 kernels (causes unstable training on MPS as of torch 2.6.0)
- Reduced token count to 40M

**CUDA (GPU):**
- Enables TF32 with `torch.set_float32_matmul_precision('high')` for faster matmul
- Full 327M token training
- Optional: Can use `torch.compile(model)` with Inductor backend

### Hyperparameters to Tune
You should find good defaults for:
- Learning rate (most important!)
- Learning rate warmup steps
- AdamW parameters (β₁, β₂, ε)
- Weight decay

### Default Starting Values
The script uses these defaults (based on common practices):
- **Learning rate**: 3e-4
- **Warmup steps**: 500
- **Weight decay**: 0.1
- **β₁**: 0.9
- **β₂**: 0.999
- **ε**: 1e-8

## Learning Rate Sweep

The `task_7_2_training.py` script automatically tries multiple learning rates:
- 1e-4
- 3e-4
- 5e-4
- 1e-3

You can modify these in the script to explore other values.

## Expected Runtime

**GPU (H100):**
- Single training run: 30-40 minutes
- Full 4-LR sweep: ~2-3 hours

**CPU (M3 Max, 36GB RAM):**
- Single training run: ~1 hour 22 minutes
- Full 4-LR sweep: ~5-6 hours

**MPS (M3 Max):**
- Single training run: ~36 minutes
- Full 4-LR sweep: ~2.5 hours

**Note:** The script automatically detects your device and adjusts:
- Token count (40M for CPU/MPS, 327M for GPU)
- Target validation loss (2.00 for CPU/MPS, 1.45 for GPU)
- Compilation strategy (device-appropriate torch.compile settings)

If your runtime is significantly longer, check:
- Data loading efficiency (use memory mapping)
- Batching is properly implemented
- Validation isn't running too frequently
- No unnecessary checkpointing

## Output

The training script produces:
1. **Checkpoints**: Saved in `models/` directory
   - Format: `task_7_2_lr_{learning_rate}.pt`
   - Contains model state, optimizer state, and hyperparameters

2. **Learning Curves**: `learning_curves_task_7_2.png`
   - Training loss vs steps
   - Validation loss vs steps
   - Comparison across learning rates

3. **Console Output**:
   - Training progress (loss, learning rate, tokens/sec)
   - Validation loss every 1000 steps
   - Final summary of all runs

## Deliverables for Task 7.2

According to the assignment, you need to provide:

1. **Learning curves** for multiple learning rates
2. **Explanation** of your hyperparameter search strategy
3. **A trained model** with validation loss (per-token) ≤ 1.45

The `task_7_2_training.py` script generates all of these automatically.

## Tips for Hyperparameter Tuning

### Learning Rate Search Strategy

1. **Start with a coarse search**: Try learning rates spanning multiple orders of magnitude (1e-5, 1e-4, 1e-3, 1e-2)

2. **Identify the range**: Look for:
   - Learning rates that cause divergence (too high)
   - Learning rates that train too slowly (too low)
   - The "sweet spot" in between

3. **Fine-grained search**: Once you find a promising range, try more values in that range

4. **Monitor both training and validation loss**: 
   - Training loss shows if the model can learn
   - Validation loss shows if it generalizes

### Other Hyperparameters

Once you have a good learning rate:
- **Warmup steps**: Try 0.01-0.05 of total steps
- **Weight decay**: Try 0, 0.01, 0.1
- **Batch size**: Larger batches often need higher learning rates

## Example Usage

### Quick Test (1 minute)
```bash
python examples/quick_demo.py
```

### Full Training (2-3 hours on GPU)
```bash
python examples/task_7_2_training.py
```

### Custom Learning Rate Sweep
Edit `task_7_2_training.py` and modify:
```python
learning_rates = [1e-4, 3e-4, 5e-4, 1e-3]  # Your custom values
```

## Troubleshooting

### Import Errors
Make sure you're running from the project root and the package is installed:
```bash
pip install -e .
```

### Missing Data Files
The tokenizer and data files should be in:
- `cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_tokens_u16.bin`
- `cs336_basics/tokenizer/TinyStoriesV2_GPT4_valid_tokens_u16.bin`
- `cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_vocab_vocab_size_10000_num_docs_2413403.pkl`
- `cs336_basics/tokenizer/TinyStoriesV2_GPT4_train_merges_vocab_size_10000_num_docs_2413403.pkl`

### Matplotlib Not Installed
The plotting is optional. Install with:
```bash
pip install matplotlib
```

### Out of Memory
If you run out of GPU memory:
- Reduce batch size (adjust steps accordingly to maintain total tokens)
- Reduce context length temporarily
- Use gradient accumulation

## Good Luck!

Remember: The goal is to achieve validation loss ≤ 1.45 on TinyStories.

