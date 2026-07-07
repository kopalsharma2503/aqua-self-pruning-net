"""SGD with momentum -- the minimum-acceptable optimizer per the spec.
Kept for comparison; Adam (optim/adam.py) is used for Part 3.

Same masked-state discipline as Adam: momentum buffers are reset to
zero at exactly the positions whose mask just changed, so a revived
connection's velocity isn't contaminated by momentum accumulated from
a completely different point in training.
"""
from __future__ import annotations

import numpy as np

from engine.tensor import Tensor


class SGD:
    def __init__(self, params: list[Tensor], lr: float = 1e-2, momentum: float = 0.9, weight_decay: float = 0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {id(p): np.zeros_like(p.data) for p in params}

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            key = id(p)
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            v = self.velocity[key]
            v[:] = self.momentum * v + g
            p.data -= self.lr * v

    def reset_state_at(self, param: Tensor, changed_mask: np.ndarray):
        key = id(param)
        if key not in self.velocity:
            return
        self.velocity[key][changed_mask] = 0.0
