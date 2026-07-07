"""Shared helpers used by all three train/ entry-point scripts."""
from __future__ import annotations

import numpy as np

from engine.tensor import Tensor, softmax_probs
from nn.layers import MLP

HIDDEN_SIZES = [64, 128, 64, 10]  # 64-dim digit input -> 10 classes


def build_model(seed: int) -> MLP:
    rng = np.random.default_rng(seed)
    return MLP(HIDDEN_SIZES, rng)


def evaluate(model: MLP, X: np.ndarray, y: np.ndarray, batch_size: int = 256):
    correct, total, loss_sum = 0, 0, 0.0
    for start in range(0, len(X), batch_size):
        xb = X[start:start + batch_size]
        yb = y[start:start + batch_size]
        logits = model(Tensor(xb))
        probs = softmax_probs(logits.data)
        preds = probs.argmax(axis=1)
        correct += (preds == yb).sum()
        total += len(yb)
        shifted = logits.data - logits.data.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        loss_sum += -log_probs[np.arange(len(yb)), yb].sum()
    return loss_sum / total, correct / total
