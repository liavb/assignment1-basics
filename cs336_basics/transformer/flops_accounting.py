"""
FLOP Accounting for Transformer Language Model

This module provides utilities to compute the number of floating-point operations (FLOPs)
required for a forward pass through a Transformer language model.

The vast majority of FLOPs in a Transformer are matrix multiplications, so we focus on:
1. Identifying all matrix multiplies in a Transformer forward pass
2. Converting each matrix multiply into FLOPs using the rule:
   For A ∈ R^(m×n) and B ∈ R^(n×p), the matrix-matrix product AB requires 2mnp FLOPs
"""

from typing import Dict, Any
import torch
from dataclasses import dataclass


@dataclass
class FLOPBreakdown:
    """Breakdown of FLOPs by component."""
    embedding: int
    attention: int
    feedforward: int
    output_projection: int
    total: int
    # Add parameter counts
    embedding_params: int
    attention_params: int
    feedforward_params: int
    output_projection_params: int
    norm_params: int
    total_params: int
    # Memory requirement (in bytes)
    memory_bytes: int

    def __str__(self) -> str:
        # Convert memory to more readable units
        memory_mb = self.memory_bytes / (1024 * 1024)
        memory_gb = memory_mb / 1024

        if memory_gb >= 1.0:
            memory_str = f"{memory_gb:.2f} GB"
        else:
            memory_str = f"{memory_mb:.1f} MB"

        return f"""FLOP Breakdown:
  Embedding:         {self.embedding:,}
  Attention:         {self.attention:,}
  Feed-Forward:      {self.feedforward:,}
  Output Projection: {self.output_projection:,}
  Total:             {self.total:,}

Parameter Count:
  Embedding:         {self.embedding_params:,}
  Attention:         {self.attention_params:,}
  Feed-Forward:      {self.feedforward_params:,}
  Output Projection: {self.output_projection_params:,}
  Normalization:     {self.norm_params:,}
  Total:             {self.total_params:,}

Memory Requirement (FP32): {memory_str} ({self.memory_bytes:,} bytes)"""


def count_matmul_flops(m: int, n: int, p: int) -> int:
    """
    Count FLOPs for matrix multiplication A @ B where A is m×n and B is n×p.
    
    Args:
        m: Number of rows in matrix A
        n: Number of columns in A / rows in B
        p: Number of columns in matrix B
        
    Returns:
        Number of FLOPs (2mnp)
    """
    return 2 * m * n * p


def count_embedding_flops(vocab_size: int, d_model: int, sequence_length: int, batch_size: int) -> int:
    """
    Count FLOPs for embedding lookup.
    
    Embedding is typically implemented as a lookup operation, which doesn't involve
    matrix multiplication. However, it can be viewed as a sparse matrix multiplication.
    For accounting purposes, we'll count it as 0 FLOPs since it's just indexing.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        sequence_length: Length of input sequence
        batch_size: Batch size
        
    Returns:
        Number of FLOPs (0 for lookup operations)
    """
    # Embedding is just lookup/indexing, no matrix multiplication
    return 0


def count_attention_flops(d_model: int, num_heads: int, sequence_length: int, batch_size: int) -> int:
    """
    Count FLOPs for multi-head self-attention.
    
    Multi-head attention involves:
    1. Q, K, V projections: 3 × (batch_size × seq_len × d_model) @ (d_model × d_model)
    2. Attention scores: (batch_size × num_heads × seq_len × d_k) @ (batch_size × num_heads × d_k × seq_len)
    3. Attention output: (batch_size × num_heads × seq_len × seq_len) @ (batch_size × num_heads × seq_len × d_k)
    4. Output projection: (batch_size × seq_len × d_model) @ (d_model × d_model)
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        sequence_length: Length of input sequence
        batch_size: Batch size
        
    Returns:
        Number of FLOPs for attention
    """
    d_k = d_model // num_heads
    flops = 0
    
    # 1. Q, K, V projections: 3 matrix multiplications
    # Input: (batch_size, seq_len, d_model) @ (d_model, d_model)
    qkv_flops = 3 * count_matmul_flops(batch_size * sequence_length, d_model, d_model)
    flops += qkv_flops
    
    # 2. Attention scores: Q @ K^T for each head
    # (batch_size, num_heads, seq_len, d_k) @ (batch_size, num_heads, d_k, seq_len)
    # This is batch_size * num_heads separate (seq_len, d_k) @ (d_k, seq_len) multiplications
    attention_scores_flops = batch_size * num_heads * count_matmul_flops(sequence_length, d_k, sequence_length)
    flops += attention_scores_flops
    
    # 3. Attention output: Attention @ V for each head
    # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, d_k)
    # This is batch_size * num_heads separate (seq_len, seq_len) @ (seq_len, d_k) multiplications
    attention_output_flops = batch_size * num_heads * count_matmul_flops(sequence_length, sequence_length, d_k)
    flops += attention_output_flops
    
    # 4. Output projection
    # (batch_size, seq_len, d_model) @ (d_model, d_model)
    output_proj_flops = count_matmul_flops(batch_size * sequence_length, d_model, d_model)
    flops += output_proj_flops
    
    return flops


def count_feedforward_flops(d_model: int, d_ff: int, sequence_length: int, batch_size: int) -> int:
    """
    Count FLOPs for SwiGLU feed-forward network.
    
    SwiGLU involves:
    1. W1 projection: (batch_size × seq_len × d_model) @ (d_model × d_ff)
    2. W3 projection: (batch_size × seq_len × d_model) @ (d_model × d_ff)  
    3. W2 projection: (batch_size × seq_len × d_ff) @ (d_ff × d_model)
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        sequence_length: Length of input sequence
        batch_size: Batch size
        
    Returns:
        Number of FLOPs for feed-forward network
    """
    flops = 0
    
    # 1. W1 projection (for SiLU activation)
    # (batch_size, seq_len, d_model) @ (d_model, d_ff)
    w1_flops = count_matmul_flops(batch_size * sequence_length, d_model, d_ff)
    flops += w1_flops
    
    # 2. W3 projection (for gating)
    # (batch_size, seq_len, d_model) @ (d_model, d_ff)
    w3_flops = count_matmul_flops(batch_size * sequence_length, d_model, d_ff)
    flops += w3_flops
    
    # 3. W2 projection (back to d_model)
    # (batch_size, seq_len, d_ff) @ (d_ff, d_model)
    w2_flops = count_matmul_flops(batch_size * sequence_length, d_ff, d_model)
    flops += w2_flops
    
    return flops


def count_output_projection_flops(d_model: int, vocab_size: int, sequence_length: int, batch_size: int) -> int:
    """
    Count FLOPs for final output projection to vocabulary.
    
    Args:
        d_model: Model dimension
        vocab_size: Size of vocabulary
        sequence_length: Length of input sequence
        batch_size: Batch size
        
    Returns:
        Number of FLOPs for output projection
    """
    # Final linear layer: (batch_size, seq_len, d_model) @ (d_model, vocab_size)
    return count_matmul_flops(batch_size * sequence_length, d_model, vocab_size)


def count_embedding_params(vocab_size: int, d_model: int) -> int:
    """Count parameters in embedding layer."""
    return vocab_size * d_model


def count_attention_params(d_model: int, num_heads: int) -> int:
    """Count parameters in multi-head attention layer."""
    # Q, K, V projections: each is d_model × d_model
    # Output projection: d_model × d_model
    return 4 * d_model * d_model


def count_feedforward_params(d_model: int, d_ff: int) -> int:
    """Count parameters in SwiGLU feed-forward layer."""
    # W1: d_model → d_ff
    # W2: d_ff → d_model
    # W3: d_model → d_ff
    return d_model * d_ff + d_ff * d_model + d_model * d_ff


def count_norm_params(d_model: int, num_layers: int) -> int:
    """Count parameters in RMSNorm layers."""
    # Each transformer block has 2 RMSNorm layers (norm1, norm2)
    # Plus 1 final output norm
    # Each RMSNorm has d_model parameters (the 'g' scaling vector)
    return (2 * num_layers + 1) * d_model


def count_output_projection_params(d_model: int, vocab_size: int) -> int:
    """Count parameters in final output projection."""
    return d_model * vocab_size


def calculate_memory_fp32(total_params: int) -> int:
    """Calculate memory required for FP32 parameters in bytes."""
    # Each FP32 parameter requires 4 bytes
    return total_params * 4


def count_transformer_lm_flops(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    batch_size: int = 1
) -> FLOPBreakdown:
    """
    Count total FLOPs for a complete TransformerLM forward pass.
    
    Args:
        vocab_size: Size of vocabulary
        context_length: Sequence length being processed
        num_layers: Number of transformer layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        batch_size: Batch size (default: 1)
        
    Returns:
        FLOPBreakdown object with detailed breakdown
    """
    # 1. Embedding FLOPs (typically 0 for lookup)
    embedding_flops = count_embedding_flops(vocab_size, d_model, context_length, batch_size)

    # 2. Attention FLOPs (per layer × num_layers)
    attention_flops_per_layer = count_attention_flops(d_model, num_heads, context_length, batch_size)
    total_attention_flops = attention_flops_per_layer * num_layers
    
    # 3. Feed-forward FLOPs (per layer × num_layers)
    ff_flops_per_layer = count_feedforward_flops(d_model, d_ff, context_length, batch_size)
    total_ff_flops = ff_flops_per_layer * num_layers
    
    # 4. Output projection FLOPs
    output_proj_flops = count_output_projection_flops(d_model, vocab_size, context_length, batch_size)

    # 5. Total FLOPs
    total_flops = embedding_flops + total_attention_flops + total_ff_flops + output_proj_flops
    
    # Parameter counts
    embedding_params = count_embedding_params(vocab_size, d_model)
    attention_params_per_layer = count_attention_params(d_model, num_heads)
    total_attention_params = attention_params_per_layer * num_layers
    feedforward_params_per_layer = count_feedforward_params(d_model, d_ff)
    total_feedforward_params = feedforward_params_per_layer * num_layers
    output_projection_params = count_output_projection_params(d_model, vocab_size)
    norm_params = count_norm_params(d_model, num_layers)

    # Total parameter count
    total_params = (
        embedding_params + total_attention_params +
        total_feedforward_params + output_projection_params + norm_params
    )

    # Memory estimation for model parameters only (FP32)
    memory_bytes = calculate_memory_fp32(total_params)

    return FLOPBreakdown(
        embedding=embedding_flops,
        attention=total_attention_flops,
        feedforward=total_ff_flops,
        output_projection=output_proj_flops,
        total=total_flops,
        embedding_params=embedding_params,
        attention_params=total_attention_params,
        feedforward_params=total_feedforward_params,
        output_projection_params=output_projection_params,
        norm_params=norm_params,
        total_params=total_params,
        memory_bytes=memory_bytes
    )


def analyze_transformer_flops(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    batch_size: int = 1,
    verbose: bool = True
) -> FLOPBreakdown:
    """
    Analyze and print FLOP breakdown for a TransformerLM.
    
    Args:
        vocab_size: Size of vocabulary
        context_length: Sequence length being processed
        num_layers: Number of transformer layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        batch_size: Batch size (default: 1)
        verbose: Whether to print detailed breakdown (default: True)
        
    Returns:
        FLOPBreakdown object with detailed breakdown
    """
    breakdown = count_transformer_lm_flops(
        vocab_size, context_length, num_layers, d_model, 
        num_heads, d_ff, batch_size
    )
    
    if verbose:
        print(f"TransformerLM FLOP Analysis")
        print(f"=" * 50)
        print(f"Model Configuration:")
        print(f"  Vocabulary Size: {vocab_size:,}")
        print(f"  Context Length: {context_length}")
        print(f"  Number of Layers: {num_layers}")
        print(f"  Model Dimension: {d_model}")
        print(f"  Number of Heads: {num_heads}")
        print(f"  Feed-Forward Dimension: {d_ff}")
        print(f"  Batch Size: {batch_size}")
        print()
        print(breakdown)
        print()
        
        # Per-component percentages
        if breakdown.total > 0:
            embedding_pct = 100 * breakdown.embedding / breakdown.total
            attention_pct = 100 * breakdown.attention / breakdown.total
            feedforward_pct = 100 * breakdown.feedforward / breakdown.total
            output_proj_pct = 100 * breakdown.output_projection / breakdown.total

            print("FLOP Percentage Breakdown:")
            print(f"  Embedding:         {embedding_pct:.1f}%")
            print(f"  Attention:         {attention_pct:.1f}%")
            print(f"  Feed-Forward:      {feedforward_pct:.1f}%")
            print(f"  Output Projection: {output_proj_pct:.1f}%")
            print(f"  Total:             {embedding_pct + attention_pct + feedforward_pct + output_proj_pct:.1f}%")

    return breakdown


# Example usage and testing
if __name__ == "__main__":

    # print('GPT-2-small FLOP Analysis')
    # gpt2_small_breakdown = analyze_transformer_flops(
    #     vocab_size=50257,
    #     context_length=1024,
    #     num_layers=12,
    #     d_model=768,
    #     num_heads=12,
    #     d_ff=6400,
    #     batch_size=1
    # )
    #
    # print('GPT-2-medium FLOP Analysis')
    # gpt2_medium_breakdown = analyze_transformer_flops(
    #     vocab_size=50257,
    #     context_length=1024,
    #     num_layers=24,
    #     d_model=1024,
    #     num_heads=16,
    #     d_ff=6400,
    #     batch_size=1
    # )
    #
    # print('GPT-2-large FLOP Analysis')
    # gpt2_large_breakdown = analyze_transformer_flops(
    #     vocab_size=50257,
    #     context_length=1024,
    #     num_layers=36,
    #     d_model=1280,
    #     num_heads=20,
    #     d_ff=6400,
    #     batch_size=1
    # )

    print('GPT-2-XL FLOP Analysis')
    gpt2_xl_breakdown = analyze_transformer_flops(
        vocab_size=50257,
        context_length=1024,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=6400,
        batch_size=1
    )

    print('GPT-2-XL-16384 FLOP Analysis')
    gpt2_xl_breakdown = analyze_transformer_flops(
        vocab_size=50257,
        context_length=16384,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=6400,
        batch_size=1
    )

    # # Example: Small transformer configuration
    # print("Example 1: Small Transformer")
    # small_breakdown = analyze_transformer_flops(
    #     vocab_size=10000,
    #     context_length=128,
    #     num_layers=6,
    #     d_model=512,
    #     num_heads=8,
    #     d_ff=2048,
    #     batch_size=1
    # )
    #
    # print("\n" + "="*80 + "\n")
    #
    # # Example: Larger transformer configuration
    # print("Example 2: Larger Transformer")
    # large_breakdown = analyze_transformer_flops(
    #     vocab_size=50000,
    #     context_length=512,
    #     num_layers=12,
    #     d_model=768,
    #     num_heads=12,
    #     d_ff=3072,
    #     batch_size=4
    # )
