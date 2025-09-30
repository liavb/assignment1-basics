import torch
from einops import einsum
from torch import nn as nn


def crossEntropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
     Compute average cross entropy loss using numerically stable log-softmax,
     while (as required) invoking the custom softmax function.

     We avoid log(softmax(target)) directly (which can be -inf if the target
     probability underflows) by:
       1. Getting probabilities via softmax (numerical stability handled there).
       2. Using the max-logit class probability p_max to recover log_sum_exp:
            p_max = 1 / sum_exp  =>  log_sum_exp = -log(p_max)
       3. log_softmax(target) = (logit_target - max_logit) - log_sum_exp

     This matches PyTorch's stable F.cross_entropy even when logits are scaled large.
    """
    # logits: (batch_size, vocab_size); targets: (batch_size,)
    batch_size, vocab_size = logits.shape
    losses = []
    for i in range(batch_size):
        li = logits[i]
        ti = targets[i]
        max_logit = torch.max(li)
        shifted_target = li[ti] - max_logit  # (logit_target - max_logit)
        # Call custom softmax (required). Shape (1, vocab_size).
        probs = softmax(li.unsqueeze(0), -1)
        # Probability of max logit (could be multiple maxima; any gives same  log_sum_exp derivation).
        # Pick first occurrence of max for stability.
        max_indices = (li == max_logit).nonzero(as_tuple=True)[0]
        p_max = probs[0, max_indices[0]]  # p_max = 1 / sum_exp
        # Recover log_sum_exp safely (p_max ~ 1 if unique max, never underflows).
        log_sum_exp = -torch.log(p_max)
        log_softmax_target = shifted_target - log_sum_exp
        losses.append(-log_softmax_target)
    return torch.stack(losses).mean()

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

    weights = softmax(scores, -1)
    out = einsum(weights, V, "... n m, ... m d_v -> ... n d_v")
    return out


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
