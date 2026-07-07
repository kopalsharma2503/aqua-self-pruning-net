"""Finite-difference gradient checking for the autodiff engine."""
from __future__ import annotations

import numpy as np

from engine.tensor import Tensor


def numerical_gradient(f, x: np.ndarray, h: float = 1e-5, n_samples: int | None = None, seed: int = 0):
    """
    Central-difference numerical gradient of scalar function f(x) -> float
    w.r.t. every entry of x (or a random subset of entries, for speed, if
    n_samples is given). Returns a full-shaped array with zeros at any
    entry that was not sampled.
    """
    rng = np.random.default_rng(seed)
    grad = np.zeros_like(x, dtype=np.float64)
    flat_idx = np.arange(x.size)
    if n_samples is not None and n_samples < x.size:
        flat_idx = rng.choice(flat_idx, size=n_samples, replace=False)

    flat_x = x.reshape(-1)
    flat_grad = grad.reshape(-1)
    for i in flat_idx:
        orig = flat_x[i]
        flat_x[i] = orig + h
        f_plus = f()
        flat_x[i] = orig - h
        f_minus = f()
        flat_x[i] = orig
        flat_grad[i] = (f_plus - f_minus) / (2 * h)
    return grad


def check_gradient(build_fn, tensors, h=1e-5, n_samples=30, rtol=1e-3, atol=1e-5, seed=0):
    """
    build_fn: () -> scalar Tensor, using closures over `tensors` (list of Tensor)
    tensors: list of Tensor objects (leaves) whose .data will be perturbed
             and whose .grad will be checked.
    Returns list of (max_abs_err, max_rel_err) per tensor, and raises
    AssertionError with a diagnostic message if any check fails.
    """
    for t in tensors:
        t.zero_grad()
    out = build_fn()
    out.backward()

    results = []
    for t in tensors:
        analytic = t.grad.copy()

        def f():
            return build_fn().data.reshape(-1)[0] if False else float(build_fn().data)

        numeric = numerical_gradient(f, t.data, h=h, n_samples=n_samples, seed=seed)
        # only compare sampled entries (others are exactly 0 in `numeric` by construction,
        # but we only assert on the ones we actually sampled to keep this fast)
        mask = numeric != 0.0
        if not mask.any():
            # extremely small chance every random sample landed on a true-zero-grad
            # entry; fall back to comparing everything to stay meaningful.
            mask = np.ones_like(numeric, dtype=bool)
        diff = np.abs(analytic - numeric)
        denom = np.maximum(np.abs(analytic), np.abs(numeric)) + 1e-8
        rel = diff / denom
        max_abs = diff[mask].max()
        max_rel = rel[mask].max()
        results.append((max_abs, max_rel))
        assert max_abs < atol or max_rel < rtol, (
            f"Gradient check failed for tensor shape={t.shape}: "
            f"max_abs_err={max_abs:.3e}, max_rel_err={max_rel:.3e}"
        )
    return results
