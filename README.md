# The Self-Pruning Network

A from-scratch (NumPy-only) reverse-mode autodiff engine, MLP, and Adam
optimizer, used to train a classifier that prunes its own least-useful
connections during training under a hard sparsity budget -- built for
the AQUA "Self-Pruning Network" take-home challenge.

**scikit-learn is used in exactly one place** (`train/data.py`, to call
`sklearn.datasets.load_digits()`) purely to load a bundled standard
dataset. No sklearn model, transformer, metric, or gradient utility is
used anywhere. `scipy.sparse` is used in exactly one other place
(`train/cost_measurement.py`) to build a genuinely sparse-aware forward
pass for the Part 4 cost measurement -- not for training. Every
tensor, op, backward pass, layer, optimizer, and pruning routine is our
own code (`engine/`, `nn/`, `optim/`, `prune/`).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(No `uv` lockfile -- plain `venv` + `requirements.txt` with pinned
versions, tested on Python 3.13.)

## Reproduce everything

All scripts fix their random seeds; numbers below were produced by
these exact commands.

```bash
# Part 1: gradient-check suite + masked-weight correctness suite
python -m pytest tests/ -v

# Part 2: train the dense baseline, save learning curve
python -m train.train_baseline --epochs 60
# -> results/part2_learning_curve.{csv,png}

# Part 3: train with self-pruning to a target sparsity
python -m train.train_prune --criterion saliency --final-sparsity 0.9 --epochs 60
python -m train.train_prune --criterion magnitude --final-sparsity 0.9 --epochs 60
# bonus: allow pruned connections to regrow if they become important again
python -m train.train_prune --criterion saliency --final-sparsity 0.9 --epochs 60 --allow-regrowth --regrowth-fraction 0.05
# -> results/part3_prune_curve_<criterion>_<sparsity>.{csv,png}

# Part 4: sparsity-accuracy Pareto sweep (5 seeds x 2 criteria x 6 sparsity levels)
python -m train.pareto_sweep
# -> results/part4_pareto_raw.csv, part4_pareto_summary.csv, part4_pareto_curve.png,
#    part4_falsifiable_claim.txt

# Part 4: honest FLOP / active-parameter / wall-clock cost measurement
python -m train.cost_measurement
# -> results/part4_cost_synthetic.csv, part4_cost_trained_model.csv
```

## Repo layout

```
engine/   reverse-mode autodiff: Tensor, ops, backward(), gradcheck
nn/       Linear layer (with an in-graph prunable mask), MLP, init
optim/    Adam and SGD-with-momentum, both mask/revival-aware
prune/    importance criteria, cubic sparsity schedule, Pruner orchestrator
train/    data loading, training loops, Pareto sweep, cost measurement
tests/    gradcheck suite, masked-weight correctness suite, pruner tests
results/  committed plots + raw CSV numbers for Parts 2-4
DESIGN.md derivations and the four required design-question answers
```

## Part 1 -- autodiff engine

`engine/tensor.py` implements a `Tensor` wrapping a NumPy array with
`+ - * /`, `matmul`, `sum`/`mean` (including reduction over an axis),
`relu`/`tanh`/`sigmoid`/`gelu`, and a fused, numerically-stable
`softmax_cross_entropy`. `backward()` does a DFS topological sort and
accumulates gradients in reverse order, so a tensor used more than once
in the graph (e.g. a weight matrix multiplied by both itself and a
mask) gets fully-summed gradients before propagating further upstream.
Every broadcasting op (`(N, D) + (D,)`, etc.) routes its gradient
through `_unbroadcast`, which sums out exactly the axes NumPy stretched
-- this is the single most common autodiff bug to get wrong, and it's
covered by an explicit test (`tests/test_gradcheck.py::test_add_broadcast_bias`).

`tests/test_gradcheck.py` checks every op against a central-difference
numerical gradient (`engine/gradcheck.py`), including a deep,
multi-layer composed graph. `python -m pytest tests/test_gradcheck.py -v`
-> 18/18 pass.

### The masked-weight correctness requirement

`tests/test_masking.py` is the test suite for the specific correctness
requirement the challenge calls out. It proves, with assertions (not
just training curves that "look fine"):

1. A masked weight contributes **exactly** `0.0` to the forward pass
   and receives **exactly** `0.0` gradient (not "small" -- checked with
   `==`), because the mask multiply is a real node in the autodiff
   graph (see `nn/layers.py::Linear.forward`).
2. A masked weight survives 20 Adam steps *with weight decay enabled*
   without drifting off exact zero (weight decay pulls every raw
   weight toward 0 using its *value*, not its gradient, so it would
   otherwise leak a masked weight away from 0 every step).
3. **The one the case study is hinting at.** After many steps pruned,
   a revived connection's first post-revival Adam update is
   numerically identical to a brand-new Adam optimizer's first update
   on a freshly initialized parameter -- not corrupted by stale
   momentum, and not blown up by a stale global step counter's decayed
   bias correction. See DESIGN.md Q2 and `optim/adam.py`'s docstring
   for the full derivation of why a naive "just zero `m` and `v`, keep
   the global `t`" fix is *still* subtly wrong, and why we use
   per-element step counters instead.

## Part 2 -- training

`nn/init.py` uses He-normal initialization (`std = sqrt(2/fan_in)`),
justified by the ReLU-halves-variance argument (see the module
docstring). `optim/adam.py` and `optim/sgd.py` are both implemented
from scratch. Trained on `sklearn.datasets.load_digits` (1797 8x8
grayscale digit images, 10 classes) with a `[64, 128, 64, 10]` MLP,
mini-batch size 32, plain Adam `lr=1e-3`, 60 epochs, seed 0:

| epoch | train_loss | train_acc | test_loss | test_acc |
|---|---|---|---|---|
| 1  | 1.61 | 0.834 | 0.83 | 0.830 |
| 10 | 0.025 | 0.999 | 0.11 | 0.967 |
| 40 | 0.001 | 1.000 | 0.08 | 0.975 |
| 60 | <0.001 | 1.000 | 0.08 | 0.975 |

No NaNs/Infs at any point (`train_baseline.py` asserts this every
step). See `results/part2_learning_curve.png`.

## Part 3 -- self-pruning

- **Importance criterion (ours): saliency = `|W * dL/dW|`**, a
  first-order Taylor approximation of the loss increase from deleting
  a connection. Full derivation in DESIGN.md Q1. Baseline: pure
  magnitude (`|W|`).
- **Schedule:** cubic sparsity ramp (Zhu & Gupta, 2017), pruning
  globally (one percentile threshold pooled across all weight
  matrices, not per-layer) every 20 steps between 5% and 60% of the
  way through training.
- **Masking:** enforced inside the autodiff graph (`W_eff = W * mask`)
  plus a hard re-zero after every optimizer step -- see Part 1 section
  above.
- **Bonus -- regrowth:** `--allow-regrowth` periodically runs a
  "dense probe" forward/backward pass with masks temporarily lifted,
  purely to read `dL/dW` at currently-pruned positions (which is
  otherwise always exactly 0 and gives no signal to rank pruned
  candidates against each other -- see `prune/pruner.py::maybe_regrow`
  docstring). The highest-probe-gradient pruned connections are
  revived and an equal number of the weakest surviving ones are cut to
  hold the sparsity budget. **Stability finding:** at a mild swap rate
  (`--regrowth-fraction 0.05`) this is stable and reaches the same
  accuracy as no-regrowth pruning
  (`results/part3_prune_curve_saliency_90_regrowth.png`). At an
  aggressive swap rate (`0.1`, i.e. swapping 10% of pruned connections
  every 3 steps) we observe a genuine, reproducible instability -- a
  transient loss spike and accuracy crash to near-chance around the
  point sparsity crosses ~85-89%, before recovering
  (`results/part3_prune_curve_saliency_90_regrowth_unstable_ablation.png`).
  We investigated this is a real dynamics effect, not a masking/gradient
  bug (Part 1's tests pass regardless): swapping too large a fraction
  of connections too frequently repeatedly perturbs the function
  non-smoothly, because the "cut" side removes currently-surviving
  connections that may still carry real signal, unlike the monotonic
  schedule which only ever removes already-lowest-saliency connections.
  See DESIGN.md for the fuller discussion.

Target 90% sparsity, seed 0: **final sparsity 0.9000, final test
accuracy 0.980** (saliency) vs. **0.983** (magnitude) -- both
comfortably above the 95%+ accuracy bar, see the honest multi-seed
comparison below for whether this particular seed's ordering is
meaningful.

## Part 4 -- evidence

### Sparsity-accuracy Pareto curve

5 seeds x 2 criteria x 6 target sparsity levels (0, 50, 75, 90, 95,
98%), `results/part4_pareto_curve.png`, raw numbers in
`results/part4_pareto_raw.csv` (60 rows) and aggregated
mean/std in `results/part4_pareto_summary.csv`:

| target sparsity | saliency mean acc (std) | magnitude mean acc (std) |
|---|---|---|
| 0%  | 97.05% (0.64pp) | 97.05% (0.64pp) |
| 50% | 97.60% (0.80pp) | 97.44% (0.77pp) |
| 75% | 97.44% (0.82pp) | 97.21% (0.79pp) |
| 90% | 97.16% (1.09pp) | 97.21% (1.04pp) |
| 95% | 96.94% (0.88pp) | 96.77% (0.61pp) |
| 98% | 95.10% (1.91pp) | 94.65% (0.50pp) |

Accuracy is essentially flat out to 90%+ sparsity and only starts
degrading meaningfully at 98%.

### Real cost measurement

`train/cost_measurement.py` measures three things and is honest about
which ones are real:

1. **Active-parameter count** (exact): e.g. the real, trained 90%-target
   saliency-pruned model's layers end up with 857/8192, 573/8192, and
   272/640 active weights respectively (`results/part4_cost_trained_model.csv`).
2. **FLOPs**: scale exactly with active connections (`2 * N * active_weights`),
   giving a real ~9.6x / ~14.3x / ~2.3x FLOP reduction per layer at
   ~90-93% per-layer sparsity.
3. **Wall-clock**, comparing three forward paths at a synthetic
   512x512 layer (`results/part4_cost_synthetic.csv`):
   - dense matmul (baseline)
   - **"dense-times-mask"**: `X @ (W * mask)` -- still a fully dense
     NumPy/BLAS matmul, just on an array that happens to contain
     zeros. **This is not a real speedup and we do not claim it as
     one**: measured at 0.35-0.44ms regardless of sparsity, identical
     to the fully dense baseline, because BLAS has no idea most entries
     are zero and performs the same multiply-adds either way.
   - **`scipy.sparse` CSR matmul**: a genuinely sparse-aware forward
     path. Honest result: at this matrix size, CSR's per-nonzero
     overhead means it's *slower* than dense NumPy below ~98% sparsity
     (10.3ms vs 0.36ms dense at 0% sparsity!) and only overtakes dense
     wall-clock time above ~99% sparsity (0.27ms vs 0.35ms). FLOP-count
     reduction is real and monotonic from the first pruned weight;
     wall-clock speedup on this naive CSR implementation is not,
     until sparsity is very high. See DESIGN.md Q3/Q4 for what changes
     this crossover point in a real serving system.

### Baseline comparison + falsifiable claim

Full text in `results/part4_falsifiable_claim.txt`:

> At 90% sparsity, saliency pruning retains 97.16% mean test accuracy
> across 5 seeds (std=1.09pp) versus 97.21% (std=1.04pp) for magnitude
> pruning -- a gap of -0.06 percentage points, **not** clearly larger
> than the pooled per-seed noise (1.51pp). At a harder 98% budget,
> saliency retains 95.10% (std=1.91pp) vs. 94.65% (std=0.50pp) for
> magnitude -- a gap of +0.45pp, again not clearly outside noise.

We report this as the real result rather than cherry-picking a seed or
sparsity level that shows a bigger gap: on a 64-input, ~17k-parameter
MLP with only 10 classes, there is enough redundant capacity that
*both* criteria can find a good enough 90-98% sparse subnetwork, and
5-seed noise (~1-2 percentage points) is comparable to or larger than
the mean difference between criteria. This is not the same as "the
criteria are equivalent" -- see DESIGN.md Q1 for why we'd still expect
saliency to pull ahead on a harder task or a much smaller sparsity
budget where the *choice* of which connections survive matters more.
