"""
Importance criteria: how does a connection earn the right to survive?

MAGNITUDE (baseline)
--------------------
score = |W|

The implicit assumption is that a small weight contributes little to
the function the network computes. This ignores the loss landscape
entirely: a small weight sitting where the loss is very sensitive to it
(large local gradient) can matter far more than a large weight sitting
in a flat region. It's the trivial answer the case study calls out.

SALIENCY (ours): first-order Taylor approximation of the loss change
from deleting a connection
----------------------------------------------------------------------
Removing connection i means setting w_i -> 0. Expand the loss around
the current weight w_i to first order:

    L(w_i + delta) ~= L(w_i) + dL/dw_i * delta

Setting delta = (0 - w_i) = -w_i (i.e. "what happens if I zero this
weight right now"):

    L(0) - L(w_i) ~= dL/dw_i * (-w_i) = -g_i * w_i

So the (signed) first-order loss change from deleting connection i is
`-g_i * w_i`, and its magnitude -- how much the loss moves, in either
direction -- is:

    saliency_i = |g_i * w_i|

A connection with LOW saliency is one the linearized loss says we can
delete for nearly free; a connection with HIGH saliency is one the loss
is predicted to move a lot if we delete it. We prune the lowest-
saliency connections, i.e. the ones the first-order model says we can
most safely remove -- exactly "loss increase if removed", to first
order.

This is the classic optimal-brain-damage-style criterion (LeCun et al.,
1989; re-popularized for modern pruning by Molchanov et al., 2017,
"Pruning Convolutional Neural Networks for Resource Efficient
Inference"). Two things worth being honest about:

1. It's a *first-order* approximation. It ignores the second-order
   (curvature) term in the Taylor expansion, which is what full Optimal
   Brain Damage / Optimal Brain Surgeon use (at the cost of needing the
   Hessian, which is not remotely tractable to compute honestly from
   scratch here). |g * w| is the practical, still-meaningfully-better-
   than-magnitude compromise.
2. A single minibatch's gradient is noisy. We reduce that noise by
   tracking an exponential moving average of |g * w| across recent
   training steps (see prune/pruner.py `update_ema`) rather than
   snapshotting the score only at the instant we prune -- a
   bias/variance trade-off: EMA smooths out minibatch noise (lower
   variance) at the cost of lagging the true instantaneous saliency by
   a few steps (some bias), which is the right trade for a slow-moving
   quantity like "is this connection still useful".
"""
import numpy as np


def magnitude_score(W: np.ndarray) -> np.ndarray:
    return np.abs(W)


def saliency_score(W: np.ndarray, grad: np.ndarray) -> np.ndarray:
    return np.abs(W * grad)
