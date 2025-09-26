from einops import einsum
from .utils import softmax


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention (section 3.5.4 notation), supporting both 3D and 4D tensors.

    Notation used here:
    - Q: ... x n x d_k  (queries)
    - K: ... x m x d_k  (keys)
    - V: ... x m x d_v  (values)

    Where:
    - n: number of queries (sequence length of queries)
    - m: number of keys/values (sequence/context length)
    - d_k: dimensionality of queries and keys
    - d_v: dimensionality of the values

    The leading "..." matches any number of batch/head dims (e.g. batch, heads).
    This implementation uses ellipsis in the einsum equations so it accepts both
    unbatched (n x d_k) and batched multi-head (batch x heads x n x d_k) inputs.
    """

    # Derive canonical dims from the trailing dimensions
    d_k = Q.shape[-1]


    # Compute attention scores with ellipsis to allow extra leading dims (batch, heads)
    scores = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / (d_k ** 0.5)

    if mask is not None:
        # mask must broadcast to shape ... x n x m; tests provide mask shaped like inputs
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = softmax(scores, -1)

    # Compute weighted sum over values
    output = einsum(attn_weights, V, "... n m, ... m d_v -> ... n d_v")
    return output
