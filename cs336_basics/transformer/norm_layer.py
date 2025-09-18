import torch
from torch import nn
from einops import einsum


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