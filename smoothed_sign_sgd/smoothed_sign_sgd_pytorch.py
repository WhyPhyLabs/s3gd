from __future__ import annotations
from typing import Callable

import torch
from torch.optim.optimizer import Optimizer

# functions

def exists(val):
    return val is not None

# update functions

def smoothed_signsgd_step(p, grad, exp_avg, lr, wd, beta):
    # stepweight decay

    p.data.mul_(1. - lr * wd)

    # weight update

    exp_avg.mul_(beta).add_(grad.sign(), alpha=1.0 - beta)

    # decay the momentum running average coefficient

    p.add_(exp_avg, alpha=-lr)

# class

class SmoothedSignSGD(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        beta: float = 0.9,
        weight_decay: float = 0.0,
        use_triton: bool = False,
        decoupled_weight_decay: bool = False,
    ):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f'β should be in [0,1); got {beta}')
        if lr <= 0.0:
            raise ValueError(f'lr must be positive; got {lr}')

        self._init_lr = lr
        self.decoupled_wd = decoupled_weight_decay

        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)

        super().__init__(params, defaults)

        self.update_fn = smoothed_signsgd_step

        if use_triton:
            from smoothed_sign_sgd.triton import smoothed_signsgd_step as triton_smoothed_signsgd_step
            self.update_fn = triton_smoothed_signsgd_step

    @torch.no_grad()
    def step(
        self,
        closure: Callable | None = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, beta, state, decoupled_wd, init_lr = p.grad, group['lr'], group['weight_decay'], group['beta'], self.state[p], self.decoupled_wd, self._init_lr

                # maybe decoupled weight decay

                if decoupled_wd:
                    wd /= init_lr

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    wd,
                    beta
                )

        return loss
