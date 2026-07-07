"""
Linear layer with a built-in prunable connection mask.

The mask is a plain (non-differentiable) NumPy 0/1 array, the same
shape as W. It participates in the forward pass as an elementwise
*multiply inside the autodiff graph*:

    W_eff = W * mask
    y     = x @ W_eff + b

Because the mask multiply is a real graph node (not a NumPy-level
side effect applied outside of autodiff), the chain rule for `mul`
automatically gives:

    dL/dW = dL/dW_eff * mask

so a masked (mask == 0) entry of W receives *exactly* zero gradient --
not a small number, not "effectively zero", but the literal float 0.0,
for free, as a consequence of correct backprop through the graph rather
than a special case bolted on afterwards. See engine/tensor.py `mul`.

We additionally hard-zero `W.data` at masked positions after every
optimizer step (`apply_mask`). This is defense in depth: it guarantees
"a masked weight must contribute zero to the forward pass" stays true
even in the presence of a weight-decay term (which pulls every weight
toward 0 using its raw value, not its gradient, and would otherwise
slowly tug a masked weight away from exact zero every step).
"""
from __future__ import annotations

import numpy as np

from engine.tensor import Tensor
from nn.init import he_normal


class Linear:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        W0 = he_normal(in_features, out_features, rng)
        self.W = Tensor(W0, requires_grad=True)
        self.b = Tensor(np.zeros(out_features), requires_grad=True) if bias else None
        # 1.0 = connection alive, 0.0 = pruned. Lives outside the autodiff
        # graph as plain data; wrapped fresh into a non-grad Tensor each
        # forward call (cheap: it's just a NumPy view, no copy).
        self.mask = np.ones_like(W0)

    def parameters(self):
        params = [self.W]
        if self.b is not None:
            params.append(self.b)
        return params

    def forward(self, x: Tensor) -> Tensor:
        mask_t = Tensor(self.mask, requires_grad=False)
        W_eff = self.W * mask_t
        out = x.matmul(W_eff)
        if self.b is not None:
            out = out + self.b
        return out

    __call__ = forward

    def apply_mask(self):
        """Hard-zero W.data at masked positions. Call after every optimizer step."""
        self.W.data *= self.mask

    # -- pruning bookkeeping -------------------------------------------------
    def set_mask(self, new_mask: np.ndarray) -> np.ndarray:
        """
        Replace the connection mask. Returns a boolean array of positions
        whose alive/pruned status *changed* (either direction), which the
        caller (the pruning loop) uses to reset optimizer state for those
        positions -- see optim/adam.py `reset_state_at`.
        """
        new_mask = new_mask.astype(self.mask.dtype)
        changed = new_mask != self.mask
        self.mask = new_mask
        self.apply_mask()
        return changed

    @property
    def n_params(self) -> int:
        return self.W.data.size

    @property
    def n_active(self) -> int:
        return int(self.mask.sum())


class ReLU:
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    __call__ = forward

    def parameters(self):
        return []


class Tanh:
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

    __call__ = forward

    def parameters(self):
        return []


class MLP:
    """A simple feed-forward classifier: [Linear -> ReLU] * (L-1) -> Linear."""

    def __init__(self, layer_sizes, rng: np.random.Generator):
        assert len(layer_sizes) >= 2
        self.linears = []
        for i in range(len(layer_sizes) - 1):
            self.linears.append(Linear(layer_sizes[i], layer_sizes[i + 1], rng))
        self.activation = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for i, layer in enumerate(self.linears):
            h = layer(h)
            if i < len(self.linears) - 1:
                h = self.activation(h)
        return h

    __call__ = forward

    def parameters(self):
        params = []
        for layer in self.linears:
            params.extend(layer.parameters())
        return params

    def weight_layers(self):
        """Linear layers whose W is eligible for pruning (all of them here;
        biases are never pruned -- they're a tiny fraction of parameters
        and zeroing them buys no compute savings since they don't
        participate in the matmul FLOP count)."""
        return self.linears

    def apply_masks(self):
        for layer in self.linears:
            layer.apply_mask()

    def sparsity_stats(self):
        total = sum(l.n_params for l in self.linears)
        active = sum(l.n_active for l in self.linears)
        return {"total": total, "active": active, "sparsity": 1.0 - active / total}
