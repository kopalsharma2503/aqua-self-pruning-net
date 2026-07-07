"""
Minimal reverse-mode autodiff engine.

Design notes
------------
Every `Tensor` produced by an op stores:
  - `_prev`: the parent tensors it was built from (edges of the graph)
  - `_backward`: a closure that knows how to push `self.grad` onto
    `_prev[i].grad` given the *local* Jacobian-vector product of the op.

`backward()` builds a topological order via DFS and calls each node's
`_backward` closure in reverse order, so every node's `.grad` is fully
accumulated (all downstream consumers have contributed) before it pushes
gradient further upstream. This is what makes the engine correct on
non-tree graphs (a tensor used more than once), which is exactly the
shape of graph we get once a weight matrix is also multiplied by a mask.

Broadcasting: NumPy silently broadcasts on ops like (N, D) + (D,). The
forward shapes then don't match the input shapes, so every backward
closure that could have broadcast its inputs runs the gradient through
`_unbroadcast`, which sums out exactly the axes NumPy would have
stretched. Getting this wrong (returning the broadcasted-shape gradient
directly) is a very common autodiff bug -- it silently breaks bias
gradients in every Linear layer, which is why we gradient-check with an
explicit broadcasting test in Part 1.
"""
from __future__ import annotations

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce `grad` (shape of the broadcasted output) back to `shape`
    (shape of the original, pre-broadcast input) by summing over the
    axes NumPy stretched."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_prev", "_backward", "_op")

    def __init__(self, data, requires_grad: bool = False, _children=(), _op: str = ""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def _ensure_grad(self):
        # Any tensor that appears as a parent of a requires_grad-producing
        # op needs a grad buffer to accumulate into, even if the user
        # never explicitly asked for it (e.g. an intermediate).
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad}, op={self._op!r})"

    # ------------------------------------------------------------------
    # graph traversal / backward
    # ------------------------------------------------------------------
    def backward(self):
        assert self.data.size == 1, "backward() only defined for scalar outputs"
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self._ensure_grad()
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    # ------------------------------------------------------------------
    # helpers shared by all binary ops that participate in autodiff
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Tensor) else Tensor(np.asarray(x, dtype=np.float64))

    def _needs_grad_with(self, other):
        return self.requires_grad or other.requires_grad

    # ------------------------------------------------------------------
    # elementwise ops
    # ------------------------------------------------------------------
    def __add__(self, other):
        other = Tensor._wrap(other)
        req = self._needs_grad_with(other)
        out = Tensor(self.data + other.data, requires_grad=req, _children=(self, other), _op="add")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,), _op="neg")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += -out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = Tensor._wrap(other)
        return self + (-other)

    def __rsub__(self, other):
        return Tensor._wrap(other) + (-self)

    def __mul__(self, other):
        other = Tensor._wrap(other)
        req = self._needs_grad_with(other)
        out = Tensor(self.data * other.data, requires_grad=req, _children=(self, other), _op="mul")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Tensor._wrap(other)
        req = self._needs_grad_with(other)
        out = Tensor(self.data / other.data, requires_grad=req, _children=(self, other), _op="div")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += _unbroadcast(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                grad_other = -out.grad * self.data / (other.data ** 2)
                other.grad += _unbroadcast(grad_other, other.data.shape)

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return Tensor._wrap(other) / self

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only scalar powers supported"
        out = Tensor(self.data ** power, requires_grad=self.requires_grad, _children=(self,), _op=f"pow{power}")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * power * (self.data ** (power - 1))

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # matmul
    # ------------------------------------------------------------------
    def matmul(self, other):
        other = Tensor._wrap(other)
        req = self._needs_grad_with(other)
        out = Tensor(self.data @ other.data, requires_grad=req, _children=(self, other), _op="matmul")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other._ensure_grad()
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    # ------------------------------------------------------------------
    # reductions
    # ------------------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis=axis)
                self.grad += np.broadcast_to(g, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        else:
            n = self.data.shape[axis]
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad, _children=(self,), _op="mean")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis=axis)
                self.grad += np.broadcast_to(g, self.data.shape) / n

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # non-linearities
    # ------------------------------------------------------------------
    def relu(self):
        mask = (self.data > 0).astype(self.data.dtype)
        out = Tensor(self.data * mask, requires_grad=self.requires_grad, _children=(self,), _op="relu")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * mask

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad, _children=(self,), _op="tanh")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * (1.0 - t ** 2)

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * s * (1.0 - s)

        out._backward = _backward
        return out

    def gelu(self):
        # tanh approximation (Hendrycks & Gimpel, 2016)
        c = np.sqrt(2.0 / np.pi)
        x = self.data
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        g = 0.5 * x * (1.0 + t)
        out = Tensor(g, requires_grad=self.requires_grad, _children=(self,), _op="gelu")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                d_inner = c * (1.0 + 3 * 0.044715 * x ** 2)
                dgelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * d_inner
                self.grad += out.grad * dgelu

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # numerically stable softmax + cross entropy, fused into one op
    # ------------------------------------------------------------------
    def softmax_cross_entropy(self, targets: np.ndarray):
        """
        self: logits, shape (N, C)
        targets: int array of shape (N,) with class indices in [0, C)
        returns a scalar Tensor: mean cross-entropy loss over the batch.

        Fusing softmax+CE (rather than chaining separate log/softmax/NLL
        ops) avoids computing exp() of large logits and log(0) of a
        near-zero softmax probability -- the classic sources of NaN in a
        hand-rolled classifier.
        """
        logits = self.data
        N = logits.shape[0]
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        log_probs = shifted - np.log(exp.sum(axis=1, keepdims=True))
        nll = -log_probs[np.arange(N), targets]
        loss_val = nll.mean()

        out = Tensor(loss_val, requires_grad=self.requires_grad, _children=(self,), _op="softmax_ce")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                grad_logits = probs.copy()
                grad_logits[np.arange(N), targets] -= 1.0
                grad_logits /= N
                self.grad += out.grad * grad_logits

        out._backward = _backward
        return out


def softmax_probs(logits: np.ndarray) -> np.ndarray:
    """Standalone stable softmax, used at eval time when we don't need grads."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
