"""
Adam from scratch, with correct handling of masked / pruned-and-revived
parameters.

THE BUG THIS FILE IS DESIGNED AROUND
-------------------------------------
The naive way to "support pruning" in Adam is: mask the forward pass,
and otherwise leave the optimizer untouched. That is wrong in two
compounding ways:

1. If gradients aren't *also* forced to zero at masked positions (e.g.
   because the mask was only applied as a NumPy side effect and not as
   a real node in the autodiff graph), a "pruned" weight keeps
   accumulating nonzero Adam momentum every step even though it
   contributes nothing to the forward pass. When later revived, its
   raw value has silently drifted and its momentum is stale relative
   to the current loss landscape.

   We avoid this at the source: `nn.layers.Linear` multiplies by the
   mask *inside* the autodiff graph, so `dL/dW` is exactly zero at
   masked positions by the chain rule (see nn/layers.py). No special
   casing is required in the optimizer for this half of the problem.

2. The second, more subtle bug: even if you *do* remember to zero out
   a revived connection's stale (m, v) momentum buffers, most
   implementations keep a single **global** step counter `t` shared by
   the whole parameter tensor for Adam's bias-correction terms
   `(1 - beta^t)`. By the time pruning starts, `t` is already large, so
   `(1 - beta^t) ≈ 1` -- bias correction is effectively a no-op. Zeroing
   `v` to 0 while `t` stays large means the *first* update after
   revival divides by `sqrt(v_hat) + eps ≈ eps`, producing a huge,
   destabilizing update -- the opposite of the "fresh start" you
   intended. Standard Adam's bias correction exists precisely to tame
   this small-`v` cold-start regime when `t` is small; using a stale
   global `t` defeats it.

   Our fix: `t` is tracked **per parameter element**, not per tensor.
   `reset_state_at` zeros `m`, `v`, AND `t` at exactly the positions
   whose mask changed. A revived connection therefore takes its first
   post-revival step with `t == 1`, identical in every respect to a
   freshly initialized Adam parameter -- correctly bias-corrected, no
   blow-up. This is verified in tests/test_masking.py.
"""
from __future__ import annotations

import numpy as np

from engine.tensor import Tensor


class Adam:
    def __init__(self, params: list[Tensor], lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {id(p): np.zeros_like(p.data) for p in params}
        self.v = {id(p): np.zeros_like(p.data) for p in params}
        # per-element step counters, NOT a single scalar -- see module docstring.
        self.t = {id(p): np.zeros_like(p.data, dtype=np.int64) for p in params}

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

            self.t[key] += 1
            t = self.t[key]
            m = self.m[key]
            v = self.v[key]
            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * (g ** 2)

            m_hat = m / (1.0 - self.beta1 ** t)
            v_hat = v / (1.0 - self.beta2 ** t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset_state_at(self, param: Tensor, changed_mask: np.ndarray):
        """
        Zero (m, v, t) at exactly the positions in `changed_mask`
        (boolean array, same shape as `param`). Called by the pruning
        loop whenever a connection's alive/pruned status flips, in
        either direction (prune OR regrow).
        """
        key = id(param)
        if key not in self.m:
            return
        self.m[key][changed_mask] = 0.0
        self.v[key][changed_mask] = 0.0
        self.t[key][changed_mask] = 0
