# Quick Start: CPU Training

## What Changed
✅ **Simplified for CPU-only**
✅ **Progress updates every 50 steps**
✅ **Only 2 learning rates** (faster to complete)
✅ **Clear progress indicators** with ETA

## Run Training

Just run this command:
```bash
python examples/task_7_2_training.py
```

## What You'll See

### Immediate Output:
```
============================================================
Task 7.2: TinyStories Training (CPU)
============================================================

Configuration:
  Device: CPU
  Total tokens: 40,960,000 (40M)
  Target loss: 2.00
  Expected time: ~1-1.5 hours per learning rate

Loading tokenizer...
✓ Vocabulary size: 10000

Loading data...
✓ Training data: 428,977,837 tokens
✓ Validation data: 49,084,969 tokens

============================================================
Learning Rate Sweep: [0.0003, 0.0005]
============================================================

############################################################
# RUN 1/2: Learning Rate = 3.00e-04
############################################################

Model parameters: 17,005,568

============================================================
STARTING TRAINING
============================================================
Compiling model for CPU (this may take a minute)...
✓ Compilation complete!

Training Configuration:
  Learning rate:     0.0003
  Batch size:        32
  Context length:    256
  Total tokens:      40,960,000
  Total steps:       5,000
  Target loss:       2.00
  Eval every:        500 steps

============================================================
TRAINING STARTED
============================================================

[  0.0%] Step     0/5000 | Loss: 9.2103 | LR: 6.00e-05 | ETA: 0.0min
[  1.0%] Step    50/5000 | Loss: 6.4521 | LR: 3.00e-04 | ETA: 82.3min
[  2.0%] Step   100/5000 | Loss: 5.8234 | LR: 3.00e-04 | ETA: 80.1min
...
```

### Every 50 Steps You See:
- **Percentage complete**: How far through training
- **Current step**: Progress out of total steps
- **Loss**: How well the model is learning
- **Learning rate**: Current LR (warmup → max → decay)
- **ETA**: Estimated time remaining

### Every 500 Steps (Evaluation):
```
>>> Evaluating at step 500...
>>> Validation loss = 4.5621
>>> Target: 2.00 (need 2.5621 improvement)

[ 10.0%] Step   500/5000 | Loss: 4.2134 | LR: 3.00e-04 | ETA: 72.5min
```

### At The End:
```
============================================================
FINAL EVALUATION
============================================================

Training completed in 82.3 minutes
Final validation loss: 1.8234
✓ SUCCESS! Target achieved (1.8234 <= 2.00)

------------------------------------------------------------
Testing text generation:
------------------------------------------------------------
Prompt:    'Once upon a time'
Generated: 'Once upon a time there was a little girl named Lily...'
```

## Settings

**CPU-Optimized:**
- **Total tokens**: 40M (instead of 327M)
- **Target loss**: 2.00 (instead of 1.45)
- **Steps**: 5,000
- **Learning rates**: [3e-4, 5e-4] (just 2 for faster testing)
- **Eval frequency**: Every 500 steps (10% of total)
- **Progress updates**: Every 50 steps

**Expected Time:**
- ~1-1.5 hours per learning rate
- ~2-3 hours total for both LRs

## What It Does

1. **Loads** tokenizer and data
2. **Compiles** model with torch.compile (speeds up CPU)
3. **Trains** first LR (3e-4) for 5000 steps
4. **Evaluates** every 500 steps
5. **Saves** checkpoint if loss ≤ 2.00
6. **Tests** text generation
7. **Repeats** for second LR (5e-4)
8. **Summarizes** results

## Files Created

- `models/task_7_2_lr_3e-04.pt` - Checkpoint (if target met)
- `models/task_7_2_lr_5e-04.pt` - Checkpoint (if target met)

## Troubleshooting

**No progress showing?**
- Check you're running the updated file
- All prints have `flush=True` now

**Too slow?**
- Normal! CPU is ~10-20x slower than GPU
- torch.compile helps but still takes time
- Expected: ~1.5 hours per LR

**Want faster testing?**
Edit line 314 to use fewer steps:
```python
learning_rates = [3e-4]  # Just 1 LR for quick test
```

Or reduce total_tokens on line 299:
```python
total_tokens = 10_000_000  # ~15 minutes
```

## That's It!

Just run:
```bash
python examples/task_7_2_training.py
```

And watch the progress bars! 🚀

