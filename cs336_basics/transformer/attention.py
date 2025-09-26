import torch
import torch.nn as nn
from einops import einsum, rearrange
from .embedding_layer import RotaryPositionalEmbedding


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None):
        """Multi-head self-attention (Section 3.5.5).

        d_model: model embedding dimension.
        num_heads: number of attention heads (d_model must be divisible by num_heads).
        max_seq_len/theta: if both provided, enable RoPE on Q and K.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # also d_v in this assignment
        # Projection weight tensors (out_features, in_features) matching provided state dict orientation
        self.q_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.o_proj = nn.Parameter(torch.empty(d_model, d_model))
        # Optional RoPE module
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
        else:
            self.rope = None
        # Init (similar to Linear layer std)
        std = (2 / (d_model + d_model)) ** 0.5
        for p in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.trunc_normal_(p, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq, d_model)
        mask: torch.Tensor | None = None,  # (batch, seq, seq) or broadcastable
        token_positions: torch.Tensor | None = None,  # (batch, seq) if RoPE
    ) -> torch.Tensor:  # (batch, seq, d_model)
        # Q,K,V projections: (batch, seq, d_model)
        # d_in = d_out = d_model
        Q = einsum(x, self.q_proj, "b s d_in, d_out d_in -> b s d_out")
        K = einsum(x, self.k_proj, "b s d_in, d_out d_in -> b s d_out")
        V = einsum(x, self.v_proj, "b s d_in, d_out d_in -> b s d_out")
        # Reshape into heads: (b, h, s, d_k)
        Q = rearrange(Q, "b s (h d_k) -> b h s d_k", h=self.num_heads)
        K = rearrange(K, "b s (h d_k) -> b h s d_k", h=self.num_heads)
        V = rearrange(V, "b s (h d_k) -> b h s d_k", h=self.num_heads)
        # Apply RoPE if enabled
        if self.rope is not None and token_positions is not None:
            pos = token_positions
            # Normalize position shape to (batch, seq)
            if pos.ndim == 1:
                pos = pos.unsqueeze(0)  # (1, seq)
            # If a single row of positions provided, expand to batch
            if pos.shape[0] == 1 and Q.shape[0] > 1:
                pos = pos.expand(Q.shape[0], -1)  # (batch, seq)
            # Expand positions to have head dimension so shapes match Q/K leading dims (batch, head, seq)
            B, H, S, D = Q.shape
            pos_heads = pos.unsqueeze(1).expand(B, H, S)  # (batch, head, seq)
            # RoPE expects x shaped (..., seq, d_k) and token_positions shaped (..., seq)
            # Our Q/K have leading dims (batch, head) which correspond to "..." - pass pos_heads accordingly
            Q = self.rope(Q, pos_heads)
            K = self.rope(K, pos_heads)
        # Attention (b, h, s, d)
        if mask is None:
            s = Q.shape[2]
            causal = torch.ones(s, s, device=Q.device, dtype=torch.bool).tril()
            mask = causal.view(1, 1, s, s)
        attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)  # (b, h, s, d_k)
        # Merge heads
        merged = rearrange(attn_out, "b h s d_k -> b s (h d_k)")  # (b, s, d_model)
        # Output projection
        # d_in = d_out = d_model
        out = einsum(merged, self.o_proj, "b s d_in, d_out d_in -> b s d_out")
        return out


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Scaled dot-product attention supporting arbitrary leading batch/head dims.

    Notation (as in assignment §3.5.4):
      Q: ... x n x d_k    (queries)
      K: ... x m x d_k    (keys)
      V: ... x m x d_v    (values)

    where n = #queries, m = #keys, d_k = key/query dim, d_v = value dim.
    The leading "..." matches any number of batch/head dims (e.g. batch, heads).
    """
    d_k = Q.shape[-1]
    # compute scores: ... x n x m
    scores = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / (d_k ** 0.5)

    if mask is not None:
        # mask expected broadcastable to (..., n, m); zero indicates masked positions
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    out = einsum(weights, V, "... n m, ... m d_v -> ... n d_v")
    return out
