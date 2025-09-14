import torch
from torch import nn
from einops import rearrange, einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None, ):
        super().__init__()
        in_features: int # final dimension of the input
        out_features: int # final dimension of the output
        device: torch.device | None = None # Device to store the parameters on
        dtype: torch.dtype | None = None # Data type of the parameters
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # Calculate standard deviation
        std = (2 / (in_features + out_features)) ** 0.5
        # Initialize weights with truncated normal distribution
        nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3 * std, b=3 * std)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the linear transformation to the input
        # (x.batch_size, x.shape_in_features), (in_features,out_features) -> x.batch_size, out_features

        return einsum(x,self.W, "batch sequence d_in, d_out d_in -> batch sequence d_out")

