"""
Weight initialization.

We use He (Kaiming) normal initialization for every Linear layer:
    W ~ N(0, 2 / fan_in)
because every hidden layer in our MLP is followed by ReLU. ReLU zeroes
out (in expectation) half of its input, so a variance-preserving init
that ignores this halving -- e.g. plain Xavier/Glorot, which assumes a
linear or symmetric activation -- systematically shrinks activation
variance layer over layer, and its gradient correspondingly vanishes
during backprop. He et al. (2015) derive the factor of 2 precisely to
cancel that halving so that activation variance (and gradient variance,
by the same argument run backward) stays approximately constant across
depth. Biases are initialized to zero, which is standard and harmless
here since weight init already breaks symmetry between units.

Empirically (see train/train_baseline.py output) this gives stable,
NaN-free training out of the box with a plain Adam learning rate of
1e-3, with no learning-rate warmup or gradient clipping required.
"""
import numpy as np


def he_normal(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(loc=0.0, scale=std, size=(fan_in, fan_out))
