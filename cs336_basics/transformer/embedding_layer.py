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