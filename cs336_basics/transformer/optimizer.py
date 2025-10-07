from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']
            m = 0 # initial value of the first moment vector
            v = 0 # initial value of the second moment vector

            for p in  group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                m = betas[0]*m + (1-betas[0]) * grad
                v = betas[1]*v + (1-betas[1]) * grad**2
                alpha_t = lr * ((math.sqrt(1-betas[1]) ** t) / (1-betas[0] ** t))
                p.data -= alpha_t * (m / (math.sqrt(v) + eps))
                p.data -= lr * weight_decay * p.data
                p.state["t"] = t + 1
                p.state["m"] = m
                p.state["v"] = v
        return loss