import math

import torch
from torch import nn
from einops import rearrange, einsum


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        # an embedding module. This function should accept the following parameters:
        super().__init__()
        num_embeddings: int # size of the vocabulary
        embedding_dim: int # dimension of the embedding vectors, i.e. d_model
        device: torch.device | None = None # device to store the parameters on
        dtype: torch.dtype | None = None # data type of the parameters
        self.embedding = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        # Initialize weights with truncated normal distribution
        nn.init.trunc_normal_(self.embedding, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Lookup the embedding vectors for the given token IDs
        embeddings = self.embedding[token_ids]
        # Use rearrange to explicitly show the dimensions
        return rearrange(embeddings, "batch sequence d_model -> batch sequence d_model")


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        r"""
           Rotary Positional Embeddings (RoPE), §3.5.3.

           Math (per position i and pair-index k ∈ {1..d/2}):
               θ_{i,k} = i · Θ^{(2k−1)/d}                                           (angle)
               For each 2D slice (x_{2k−2}, x_{2k−1}) we apply a rotation:
                   [x'_{2k−2}]   [ cosθ  −sinθ ] [x_{2k−2}]
                   [x'_{2k−1}] = [ sinθ   cosθ ] [x_{2k−1}]                         (block rotation)

           Complex-number view (same thing):
               Let z = x_even + i·x_odd. Then z' = z · e^{i θ_{i,k}}.

           Implementation idea:
             • Precompute cos(θ_{i,k}) and sin(θ_{i,k}) for all i∈[0..S-1], k∈[0..d/2−1].
             • Store them as non-trainable, device-aware BUFFERS (so they move with .to(), .half()).
             • At runtime, split channels into even/odd and apply the two elementwise formulas:

                   x_even' = x_even * cos − x_odd * sin
                   x_odd'  = x_even * sin + x_odd * cos

               No huge (S, d, d) matrices are ever built.

        """
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"
        self.theta: float = theta # Θ value for the RoPE
        self.d_k = d_k # dimension of query and key vectors
        self.max_seq_len: int = max_seq_len # Maximum sequence length that will be inputted
        self.device = device #  Device to store the buffer on

        # ---- Precompute angle table θ_{i,k0} with 0-based k0 ∈ {0..d/2−1} ----
        # NOTE: The paper/assignment uses 1-based k in (2k−1)/d. With 0-based k0, it becomes (2k0+1)/d.
        # Positions: shape [S]
        pos = torch.arange(self.max_seq_len, device=device) # i
        # Pair indices: shape [D2]
        k0 = torch.arange(self.d_k // 2, device=device)  # k0
        # Frequencies for each pair: Θ^{(2k0+1)/d}, shape [D2]
        freq = 1.0 / (self.theta ** (2 * k0 / self.d_k)) # [D2]

        # Angles: θ_{i,k0} = i * Θ^{(2k0+1)/d}, shape [S, D2]
        angles = pos[:, None] * freq[None, :]
        # Precompute cos/sin tables, shape [S, D2]
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # Store as BUFFERS (not Parameters): device/dtype-aware, non-trainable, saved in state_dict.
        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x:         (..., seq_len, d_k)
        positions: (..., seq_len)  ints in [0, max_seq_len)

        Returns:
            (..., seq_len, d_k) — RoPE-rotated x (apply to Q or K, not V).
        """
        # 1) Gather cos/sin for these specific positions: shape (..., seq_len, d_k//2)
        cos = self.cos_table[token_positions]
        sin = self.sin_table[token_positions]
        # 2) Split the channel dimension into even/odd indices (each shape: (..., seq_len, d_k//2))
        x_even = x[..., 0::2]     # channels 0,2,4,...  (real part in complex view)
        x_odd  = x[..., 1::2]     # channels 1,3,5,...  (imag part in complex view)
        # 3) Apply the 2×2 rotation per pair, vectorized:
        #    x_even' = x_even * cos − x_odd * sin
        #    x_odd'  = x_even * sin + x_odd * cos
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos
        # 4) Interleave back into (..., seq_len, d_k)
        out = torch.empty_like(x)
        out[..., 0::2] = x_rot_even
        out[..., 1::2] = x_rot_odd
        return out