import torch
from einops import einsum
from torch import nn as nn


def softmax(tensor: torch.Tensor, i: int):
    """
    A numerically stable softmax function that works on the specified dimension.
    tensor: input tensor
    i: dimension to apply softmax on
    """
    max_val = torch.max(tensor, dim=i, keepdim=True).values
    exp_tensor = torch.exp(tensor - max_val)
    sum_exp = torch.sum(exp_tensor, dim=i, keepdim=True)
    return exp_tensor / sum_exp


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


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
