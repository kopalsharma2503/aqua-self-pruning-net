"""
Pruning schedule: cubic sparsity ramp (Zhu & Gupta, 2017,
"To prune, or not to prune: exploring the efficacy of pruning for model
compression").

    s(t) = s_f + (s_i - s_f) * (1 - (t - t0) / (n * dt))^3

Gradual pruning is preferred over one-shot pruning at the end of
training for a concrete reason: one-shot pruning at target sparsity
removes a large fraction of connections in a single step, after the
surviving weights have only ever been optimized *with* the soon-to-be-
removed connections around them. The network has to re-adapt to a
suddenly very different function from a single, large perturbation,
and (especially at high sparsity, e.g. 90%+) frequently never recovers
the accuracy it had. Gradual pruning instead removes a few connections
at a time and gives the optimizer many subsequent steps to redistribute
the lost capacity onto the surviving connections before the next
increment -- the network is rarely more than a small perturbation away
from "a network it already knows how to use well".

The cubic shape (rather than e.g. linear) front-loads pruning: it's
steep just after t0 and flattens out approaching t0 + n*dt. Empirically
(Zhu & Gupta and follow-up work) this works better than a linear ramp
because early in training there is a lot of redundant/low-saliency
capacity to remove cheaply, while late in training, near the target
sparsity, the surviving connections are increasingly load-bearing and
benefit from smaller, less frequent removals plus more fine-tuning time
between them.
"""
import numpy as np


def cubic_sparsity(step: int, start_step: int, end_step: int, final_sparsity: float,
                    initial_sparsity: float = 0.0) -> float:
    if step < start_step:
        return initial_sparsity
    if step >= end_step:
        return final_sparsity
    progress = (step - start_step) / (end_step - start_step)
    return final_sparsity + (initial_sparsity - final_sparsity) * (1 - progress) ** 3
