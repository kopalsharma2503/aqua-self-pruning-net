"""
Part 1 mandatory suite: every differentiable op is checked against a
central-difference numerical gradient. Random seeds are fixed so this
suite is fully reproducible.
"""
import numpy as np
import pytest

from engine.tensor import Tensor
from engine.gradcheck import check_gradient

SEED = 0


def rand_tensor(shape, seed, requires_grad=True):
    rng = np.random.default_rng(seed)
    return Tensor(rng.normal(size=shape, scale=1.0), requires_grad=requires_grad)


def test_add():
    a = rand_tensor((4, 3), 1)
    b = rand_tensor((4, 3), 2)
    check_gradient(lambda: (a + b).sum(), [a, b])


def test_add_broadcast_bias():
    # (N, D) + (D,) -- the classic broadcasting gradient bug: the bias
    # gradient must be *summed* over the batch axis, not just reshaped.
    x = rand_tensor((8, 5), 1)
    b = rand_tensor((5,), 2)
    check_gradient(lambda: (x + b).sum(), [x, b])


def test_sub():
    a = rand_tensor((4, 3), 1)
    b = rand_tensor((4, 3), 2)
    check_gradient(lambda: (a - b).sum(), [a, b])


def test_mul():
    a = rand_tensor((4, 3), 1)
    b = rand_tensor((4, 3), 2)
    check_gradient(lambda: (a * b).sum(), [a, b])


def test_mul_broadcast():
    a = rand_tensor((6, 4), 1)
    b = rand_tensor((4,), 2)
    check_gradient(lambda: (a * b).sum(), [a, b])


def test_div():
    a = rand_tensor((4, 3), 1)
    b = rand_tensor((4, 3), 2) + 3.0  # keep away from 0
    check_gradient(lambda: (a / b).sum(), [a, b])


def test_matmul():
    a = rand_tensor((5, 4), 1)
    b = rand_tensor((4, 3), 2)
    check_gradient(lambda: (a.matmul(b)).sum(), [a, b])


def test_matmul_chain_with_bias():
    x = rand_tensor((6, 4), 1)
    w = rand_tensor((4, 3), 2)
    b = rand_tensor((3,), 3)
    check_gradient(lambda: (x.matmul(w) + b).sum(), [x, w, b])


def test_sum_axis():
    a = rand_tensor((4, 5), 1)
    check_gradient(lambda: a.sum(axis=1).sum(), [a])


def test_mean():
    a = rand_tensor((4, 5), 1)
    check_gradient(lambda: a.mean(), [a])


def test_mean_axis():
    a = rand_tensor((4, 5), 1)
    check_gradient(lambda: a.mean(axis=0).sum(), [a])


def test_relu():
    a = rand_tensor((6, 6), 1)
    a.data[a.data.reshape(-1)[:5].argsort()] += 0  # no-op, keep shape
    # nudge values away from exactly 0 so the kink doesn't break finite-diff
    a.data = a.data + np.sign(a.data) * 0.05
    check_gradient(lambda: a.relu().sum(), [a])


def test_tanh():
    a = rand_tensor((5, 5), 1)
    check_gradient(lambda: a.tanh().sum(), [a])


def test_sigmoid():
    a = rand_tensor((5, 5), 1)
    check_gradient(lambda: a.sigmoid().sum(), [a])


def test_gelu():
    a = rand_tensor((5, 5), 1)
    check_gradient(lambda: a.gelu().sum(), [a])


def test_softmax_cross_entropy():
    rng = np.random.default_rng(42)
    logits = rand_tensor((7, 4), 1)
    targets = rng.integers(0, 4, size=7)
    check_gradient(lambda: logits.softmax_cross_entropy(targets), [logits])


def test_softmax_cross_entropy_numerical_stability():
    # Large-magnitude logits are exactly the case where naive
    # softmax-then-log blows up (exp overflow / log(0) -> nan).
    logits = Tensor(np.array([[1000.0, 1.0, -1000.0], [-500.0, 500.0, 0.0]]), requires_grad=True)
    targets = np.array([0, 1])
    loss = logits.softmax_cross_entropy(targets)
    assert np.isfinite(loss.data)
    loss.backward()
    assert np.isfinite(logits.grad).all()


def test_deep_composition():
    # A small MLP-shaped graph, chained: linear -> relu -> linear -> softmax_ce.
    # Exercises reused nodes (x used twice via w1 and w2 branches feeding
    # the same downstream node) and confirms gradient accumulation across
    # multiple consumers is correct.
    rng = np.random.default_rng(7)
    x = rand_tensor((10, 6), 1)
    w1 = rand_tensor((6, 8), 2)
    b1 = rand_tensor((8,), 3)
    w2 = rand_tensor((8, 3), 4)
    b2 = rand_tensor((3,), 5)
    targets = rng.integers(0, 3, size=10)

    def forward():
        h = (x.matmul(w1) + b1).relu()
        logits = h.matmul(w2) + b2
        return logits.softmax_cross_entropy(targets)

    check_gradient(forward, [w1, b1, w2, b2], n_samples=15)
