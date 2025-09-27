import torch
from einops import einsum, rearrange
from torch import nn as nn
from cs336_basics.transformer.utils import scaled_dot_product_attention, SiLU


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None, ):
        super().__init__()
        in_features: int # final dimension of the input
        out_features: int # final dimension of the output
        device: torch.device | None = None # Device to store the parameters on
        dtype: torch.dtype | None = None # Data type of the parameters
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype, requires_grad=True)
        )
        # Calculate standard deviation
        std = (2 / (in_features + out_features)) ** 0.5
        # Initialize weights with truncated normal distribution
        nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3 * std, b=3 * std)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the linear transformation to the input
        # (x.batch_size, x.shape_in_features), (in_features,out_features) -> x.batch_size, out_features

        return einsum(x,self.W, "batch sequence d_in, d_out d_in -> batch sequence d_out")


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """Construct the RMSNorm module. This function should accept the following parameters:"""
        super().__init__()
        self.d_model: int = d_model  # Hidden dimension of the model
        self.eps: float  = eps # Epsilon value for numerical stability
        self.device: torch.device | None = device # device to store the parameters on
        self.dtype: torch.dtype | None = dtype # data type of the parameters
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape"""
        orig_dtype = x.dtype
        x_f = x.to(torch.float32)
        # Compute RMS along the last dimension (for each embedding vector): result shape is (batch, seq, 1).
        rms = x_f.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        # Use einsum to apply per-dimension scale `g` explicitly (apply the learnable parameter on each embedding vector of each token)
        scaled = einsum(x_f, self.g, 'batch_size sequence_length d_model, d_model -> batch_size sequence_length d_model')
        # step 2: divide by RMS (broadcast over last dim)
        y = scaled / rms
        return y.to(orig_dtype)


class SwigLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """Construct the SwiGLU module. This function should accept the following parameters:"""
        super().__init__()
        self.d_model: int = d_model
        self.Linear1 = Linear(in_features=d_model, out_features=d_ff) # W1: d_model -> d_ff
        # Linear2 will hold W2 (d_model x d_ff): projects back to d_model from d_ff
        self.Linear2 = Linear(in_features=d_ff, out_features=d_model)  # W2: d_ff -> d_model
        # Linear3 will hold W3 (d_ff x d_model): second projection from input to d_ff
        self.Linear3 = Linear(in_features=d_model, out_features=d_ff)  # W3: d_model -> d_ff
        self.silu = SiLU()



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape"""

        w1_matmul = self.Linear1.forward(x) # matmul(W1, x) -> (batch_size, sequence_length, d_ff)
        silu_res = self.silu.forward(w1_matmul) #  -> (batch_size, sequence_length, d_ff)
        # Compute the second projection (W3) from the input
        w3_matamul = self.Linear3.forward(x) # matmul(W3, x) -> (batch_size, sequence_length, d_ff)
        # Element-wise multiplication of the two results
        element_wise_mul = einsum(silu_res, w3_matamul, "b s d_ff, b s d_ff -> b s d_ff")
        # Project back to d_model using W2
        res = self.Linear2(element_wise_mul) # matmul(W2, element_wise_mul) -> (batch_size, sequence_length, d_model)
        return res


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


class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int | None = None,
                 theta: float | None = None):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads # Number of heads to use in multi-head self-attention
        self.d_ff = d_ff # Dimensionality of the position-wise feed-forward inner layer
        # Multi-head attention
        self.mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta)
        # Pre-norms
        self.norm1 = RMSNorm(d_model=d_model)
        self.norm2 = RMSNorm(d_model=d_model)
        # Position-wise feed-forward (SwiGLU)
        self.ffn = SwigLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm transformer block: attention then FFN, both with residuals.

        Accepts optional token_positions which will be forwarded to the MHA so RoPE
        can be applied when present.
        """
        # If RoPE is enabled in MHA, pass default token positions [0, 1, 2, ...]

        token_positions = None
        if self.mha.rope is not None:
            seq_len = x.shape[1]
            token_positions = torch.arange(seq_len, device=x.device)

        y1 = x + self.mha(self.norm1(x), token_positions=token_positions)
        y2 = y1 + self.ffn(self.norm2(y1))
        return y2