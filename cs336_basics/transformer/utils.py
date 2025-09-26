import torch


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
