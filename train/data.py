"""
Dataset loading.

scikit-learn is used ONLY here, to load a standard bundled dataset
(`sklearn.datasets.load_digits` -- 1797 8x8 grayscale digit images, 10
classes, no download required). No sklearn model, transformer, or
gradient utility is used anywhere in this repository; all training,
autodiff, and pruning code is our own NumPy implementation.

We picked digits over full MNIST so the entire Part 4 sweep (multiple
sparsity levels x multiple seeds) finishes in minutes on a laptop CPU
using our pure-Python/NumPy training loop, while still being a real,
standard, non-trivial multi-class image classification benchmark with
enough redundancy in a 64-dim input to make pruning a meaningful story
(a linear classifier alone gets ~95% on this dataset, so there is
genuine slack for an MLP to give up in exchange for sparsity).
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits


def load_digits_split(seed: int = 0, test_frac: float = 0.2):
    data = load_digits()
    X = data.data.astype(np.float64)  # (1797, 64), pixel values in [0, 16]
    y = data.target.astype(np.int64)

    X = X / 16.0  # scale to [0, 1]
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def iterate_minibatches(X, y, batch_size, rng: np.random.Generator, shuffle=True):
    n = len(X)
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        batch_idx = order[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]
