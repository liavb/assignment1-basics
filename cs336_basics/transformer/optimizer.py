from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            )

        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                t = state["t"] + 1  # increment step counter
                grad = p.grad.data
                m = state["m"]
                v = state["v"]

                m = betas[0]*m + (1-betas[0]) * grad
                v = betas[1]*v + (1-betas[1]) * grad**2
                alpha_t = lr * (math.sqrt(1-(betas[1] ** t)) / (1-(betas[0] ** t)))
                p.data -= alpha_t * (m / (torch.sqrt(v) + eps))
                p.data -= lr * weight_decay * p.data

                # Update state
                state["t"] = t
                state["m"] = m
                state["v"] = v
        return loss