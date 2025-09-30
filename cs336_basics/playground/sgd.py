from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p] # get the state associated with p
                t = state.get("t", 0) # get the iteration number from the state, or 0 in case not exist yet
                grad = p.grad.data # get the gradient of loss with respect to p
                p.data -= lr / math.sqrt(t+1) * grad # update weight tensor
                state['t'] = t + 1 # inc iteration number
        return loss

def train_loop(opt, weights, epochs: int=100):
    for t in range(epochs):
        opt.zero_grad() # reset the gradient for all learnable parameters
        loss = (weights**2).mean() # compute loss scalar value
        print(loss.cpu().item())
        loss.backward() # run the backward pass which computes gradients
        opt.step() # run the optimizer step


if __name__ == '__main__':

    lrs = [1e1, 1e2, 1e3, 1e4]
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    epochs = 10
    for lr in lrs:
        print('lr', lr)
        opt = SGD([weights], lr=lr)
        train_loop(opt, weights, epochs)

    # As we can see:
    #     lr = 10.0 (baseline)
    #     19.11343002319336
    #     12.232596397399902
    #     9.01734733581543
    #     7.055111885070801
    #     5.714639663696289
    #     4.738091945648193
    #     3.9959511756896973
    #     3.414654493331909
    #     2.9488227367401123
    #     2.5687525272369385

    # lr = 100.0 (loss decay faster)
    # 2.254102945327759
    # 2.2541027069091797
    # 0.38674288988113403
    # 0.009255627170205116
    # 1.3368213117618515e-17
    # 1.4899701936006223e-19
    # 5.017254311917586e-21
    # 2.98881297363632e-22
    # 2.563996734916404e-23
    # 2.8488853705822414e-24

    # lr = 1000 (loss decay faster in first epochs and then it goes up)
    # 3.8485296631592746e-25
    # 1.389318854645686e-22
    # 2.3995729241577663e-20
    # 2.669268202931308e-18
    # 2.1621070520548672e-16
    # 1.3645372131483244e-14
    # 7.005086614941813e-13
    # 3.013888097425088e-11
    # 1.1108549635707732e-09
    # 3.567078721289363e-08


     #lr 10000.0 (diverge!)
    # 1.01129853646853e-06
    # 0.04004843533039093
    # 789.6813354492188
    # 10347507.0
    # 101415878656.0
    # 793286633586688.0
    # 5.159826649734185e+18
    # 2.870979235074113e+22
    # 1.3951746624269228e+26
    # 6.0161483169203e+29