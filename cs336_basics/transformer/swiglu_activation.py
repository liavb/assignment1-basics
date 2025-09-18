import torch
from torch import nn
from einops import einsum
from .linear_layer import Linear

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class SwigLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """Construct the SwiGLU module. This function should accept the following parameters:"""
        super().__init__()
        self.d_model: int = d_model
        self.Linear1 = Linear(in_features=d_model, out_features=d_ff) # First linear layer
        self.Linear2 = Linear(in_features=d_model, out_features=d_ff) # Second linear layer
        self.Linear3 = Linear(in_features=d_ff, out_features=d_model) # Third linear layer
        self.silu = SiLU()



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape"""

        w1_matmul = self.Linear1.forward(x) # matmul(W1, x) -> (batch_size, sequence_length, d_ff)
        silu_res = self.silu.forward(w1_matmul) #  -> (batch_size, sequence_length, d_ff)
        w3_matamul = self.Linear3.forward(x) # matmul(W3, x) -> (batch_size, sequence_length, d_ff)
        # Element-wise multiplication of the two results
        element_wise_mul = einsum(silu_res, w3_matamul, "b s d_ff, b s d_ff -> b s d_ff")
        res = self.Linear2(element_wise_mul) # matmul(W2, element_wise_mul) -> (batch_size, sequence_length, d_model)
        return res