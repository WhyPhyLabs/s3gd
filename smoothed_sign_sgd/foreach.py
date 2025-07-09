from __future__ import annotations
from typing import Callable

import torch
from torch.optim.optimizer import Optimizer

# functions

def exists(val):
    return val is not None

# class

class SmoothedSignSGD(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        beta: float = 0.9,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False
    ):
        assert lr > 0.
        if not (0. <= beta < 1.):
            raise ValueError(f"beta must be in [0, 1); got {beta}")
        assert all([hasattr(torch, f'_foreach_{attr}_') for attr in ('mul', 'add', 'lerp')]), 'this version of torch does not have the prerequisite foreach functions'

        needed_outplace = ("sign",)
        assert all(hasattr(torch, f"_foreach_{fn}") for fn in needed_outplace)

        self._init_lr = lr
        self.decoupled_wd = decoupled_weight_decay

        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)

        super().__init__(params, defaults)

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

            lr, wd, beta, decoupled_wd, init_lr = group['lr'], group['weight_decay'], group['beta'], self.decoupled_wd, self._init_lr

            # maybe decoupled weight decay

            if decoupled_wd:
                wd /= init_lr

            # accumulate List[Tensor] for foreach inplace updates

            params = []
            grads = []
            exp_avgs = []

            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, state = p.grad, self.state[p]

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                params.append(p)
                grads.append(grad)
                exp_avgs.append(exp_avg)

            if not params:  # nothing to do in this param-group
                continue

            # stepweight decay

            if wd != 0.:
                torch._foreach_mul_(params, 1. - lr * wd)

            # weight update

            signs = torch._foreach_sign(grads)
            torch._foreach_lerp_(exp_avgs, signs, 1.0 - beta)

            torch._foreach_add_(params, exp_avgs, alpha=-lr)

        return loss
