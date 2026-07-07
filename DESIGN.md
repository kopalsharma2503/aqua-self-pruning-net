# DESIGN.md

## Q1. Derive your importance criterion and explain why it approximates the loss change from removing a connection.

**Setup.** Let `L(W)` be the training loss as a function of one weight
`w_i` (holding everything else fixed), and let `g_i = dL/dw_i` be its
gradient, both evaluated at the current point in training. "Removing"
connection `i` means setting `w_i -> 0`.

**First-order Taylor expansion.** Expand `L` around the current value
of `w_i`:

```
L(w_i + delta) ≈ L(w_i) + g_i * delta + O(delta^2)
```

Set `delta = 0 - w_i = -w_i` (the specific perturbation "delete this
connection right now"):

```
L(0) - L(w_i) ≈ g_i * (-w_i) = -g_i * w_i
```

The magnitude of this predicted loss change -- how far the loss moves,
regardless of direction -- is:

```
saliency_i = |ΔL_i| ≈ |g_i * w_i|
```

This is `prune/criteria.py::saliency_score`. A connection with **low**
saliency is one the linearized loss says can be deleted almost for
free; pruning the globally lowest-saliency connections is, to first
order, the choice that increases the loss the least for a given amount
of sparsity removed.

**Why this beats magnitude alone.** Magnitude pruning (`|w_i|`)
implicitly assumes small weight ⟹ unimportant. That assumption ignores
the loss landscape: a small weight sitting where `dL/dw_i` is large
(the loss is very sensitive to it right now, e.g. it's in the middle of
being learned, or it interacts multiplicatively with something
downstream) can matter far more than a large weight sitting in a flat
region where `dL/dw_i ≈ 0` (already converged, or dead/saturated).
`|w_i * g_i|` is exactly the quantity that captures both factors at
once: it is small only when *either* the weight is small *or* the loss
doesn't care about it -- i.e. when actually deleting it is safe.

**What it is not.** This is a first-order approximation; it drops the
`O(delta^2)` curvature term. The historically "correct" way to include
curvature is Optimal Brain Damage / Optimal Brain Surgeon (LeCun et
al., 1989; Hassibi & Stork, 1993), which uses (an approximation of) the
Hessian diagonal or inverse Hessian. We do not implement that here: an
honest Hessian-based criterion needs either Hessian-vector products
(a second backward pass through the first backward pass -- our engine
doesn't implement double-backward) or a diagonal Gauss-Newton
approximation, both of which are a meaningfully larger undertaking than
this challenge's time budget, and both would still be approximations.
`|w * g|` is the standard, defensible, one-backward-pass compromise
(Molchanov et al., 2017, use exactly this for structured/channel
pruning; we apply the same idea per-connection).

**Noise reduction.** A single minibatch's `g_i` is noisy. We track an
exponential moving average, `ema_i <- decay * ema_i + (1 - decay) *
|w_i * g_i|`, updated every training step (`Pruner.update_ema`), and
prune against the EMA rather than the instantaneous score. This is a
bias/variance trade: the EMA lags the true instantaneous saliency by
roughly `1/(1-decay)` steps (bias), in exchange for much lower variance
from minibatch sampling noise -- the right trade for "is this
connection still useful", which is a slowly-changing property, not
something we need to react to within a single batch.

**Why the Part 4 result doesn't show a big saliency-vs-magnitude gap
here, and when it would.** Our Pareto sweep (`results/part4_pareto_*`)
shows the two criteria are statistically indistinguishable (within
seed noise) on this task, even at 90-98% sparsity. We believe this is a
property of the *task*, not evidence the criteria are equivalent:
digits is a 64-dimensional, 10-class problem with a lot of redundant
capacity relative to a 3-layer, ~17k-parameter MLP (a linear classifier
alone gets >95% on this dataset). At 90-98% sparsity there are still
many different "good enough" sparse subnetworks, so both a
saliency-aware and a magnitude-only search land in that broad basin.
We would expect the gap to widen (a) on a task where fewer connections
are truly redundant (harder classification problems, or a network
sized closer to its true capacity requirement), or (b) at even more
extreme sparsity budgets (99.5%+) where there is no longer a wide
margin of "any reasonable subnetwork works" and the specific choice of
*which* connections survive starts to matter.

## Q2. What does your engine compute as "the gradient of a masked weight," and why is that the right choice?

**The forward path.** `nn/layers.py::Linear.forward` computes
`W_eff = W * mask` as a real node in the autodiff graph (`mask` is
wrapped in a `Tensor(..., requires_grad=False)`), then
`y = x @ W_eff + b`. `mask` is never applied as a NumPy-level side
effect outside the graph.

**The gradient.** By the product rule for elementwise multiply
(`engine/tensor.py::Tensor.__mul__`), `dL/dW = dL/dW_eff * mask`. At a
masked position (`mask == 0`), this is `dL/dW_eff * 0 = 0.0` --
**exactly** zero, as a direct, unconditional consequence of correctly
implemented backprop through a real graph node, not a special case we
had to remember to add. `tests/test_masking.py::test_masked_weight_forward_and_grad_are_exactly_zero`
checks this with `==`, not a tolerance.

**Why zero is the right answer**, not (say) "the gradient the weight
*would* have had if it weren't masked": the quantity `dL/dW` is, by
definition, "how much does the loss change per unit change in this
parameter's *current forward contribution*". A masked weight's current
forward contribution is fixed at 0 regardless of `W`'s stored value --
changing `W` at a masked position does not change `y` at all. So the
true derivative of the loss with respect to what the optimizer is
actually about to update is 0, full stop. Giving it any nonzero value
(e.g. "as if unmasked") would tell the optimizer to move a number that
has, by construction, no effect on the network's output -- which is
exactly the bug class the challenge is testing for.

**This is a deliberately different question from "should a pruned
connection ever be reconsidered."** Because the masked gradient is
always exactly zero, direct saliency/magnitude scores can *never*
regrow a pruned connection on their own (both `|W|` and `|W * grad|`
are stuck at 0 forever once masked). Our bonus regrowth mechanism
(`prune/pruner.py::maybe_regrow`) deliberately asks a *different*
question -- "if this connection existed, would the loss want to use
it?" -- by running a separate, throwaway forward/backward pass with the
mask temporarily lifted, and using *that* probe gradient only to *rank
candidates for revival*, never to update the real (masked) weight or
its real optimizer state. Keeping these two roles separate (real
gradient used to update parameters vs. probe gradient used only to
rank pruning/regrowth decisions) is, we think, the key discipline that
keeps "let's also do regrowth" from quietly reintroducing the exact bug
Part 1 is testing for.

**The optimizer-state half of the requirement.** Getting the gradient
right is necessary but not sufficient -- as the case study explicitly
flags, the optimizer's own state can still corrupt a revived
connection even with a perfectly correct gradient. Two failure modes,
both handled in `optim/adam.py`:

1. *Stale momentum.* If a pruned-then-revived connection's Adam
   moments (`m`, `v`) are left untouched, its first post-revival update
   is computed from momentum accumulated at a completely different
   point in training (a different loss landscape, possibly a different
   sign of what's currently useful). We reset `m` and `v` to exactly 0
   at exactly the positions whose mask just changed
   (`Adam.reset_state_at`, called from `Pruner.maybe_prune` /
   `maybe_regrow` every time `Linear.set_mask` reports a change).

2. *The subtler bug: a global step counter.* Zeroing `m` and `v` is not
   enough by itself if bias correction uses a single scalar step count
   `t` shared by the whole tensor. By the time pruning starts, `t` is
   already large, so `(1 - beta^t) ≈ 1` — bias correction is
   effectively disabled. Zeroing `v` while `t` stays large means the
   first update after revival divides by `sqrt(v_hat) + eps ≈ eps`,
   producing a spurious, oversized update instead of Adam's intended
   well-scaled cold-start step. `tests/test_masking.py::test_naive_global_step_counter_would_have_blown_up`
   works this out numerically: for standard betas, the naive
   global-`t` approach's first update is `(1 - beta1) / sqrt(1 - beta2)
   ≈ 3.16x` larger than the correct one, *independent of the gradient's
   actual magnitude*. Our fix: `Adam` tracks `t` **per parameter
   element**, not per tensor, and resets it to 0 alongside `m`, `v` at
   the changed positions. A revived connection's first update is then
   provably identical to a brand-new Adam optimizer's first update on a
   freshly initialized parameter
   (`test_optimizer_state_reset_on_revival_matches_fresh_init`), which
   is the "well-defined, not-silently-corrupting" treatment the
   challenge asks for.

## Q3. Where does your autodiff engine bottleneck, and how would you optimize it?

Profiling the training loop (`train/train_prune.py`) at the scale used
in this repo (a few thousand parameters, batches of 32) shows the
bottleneck is **not FLOPs, it's Python-level graph-construction
overhead**:

- Every op call (`+`, `*`, `matmul`, ...) allocates a new `Tensor`, a
  new `set` for `_prev`, and a new closure for `_backward`. For a
  4-layer MLP this is on the order of a few dozen Python object
  allocations per forward pass, dwarfing the actual NumPy compute time
  for small matrices (a `(32, 64) @ (64, 128)` matmul is microseconds
  of BLAS; the surrounding Python bookkeeping is comparable or larger).
- `backward()` rebuilds the topological order via DFS from scratch on
  every call (necessary for a define-by-run engine where the graph can
  differ step to step, but wasted work when, as here, the same
  architecture runs every step).
- Every `Linear.forward` re-wraps `self.mask` in a fresh `Tensor` each
  call rather than caching it, and recomputes `W * mask` as a full
  dense elementwise pass over the whole (mostly-soon-to-be-pruned)
  matrix even after most of it is zero.
- Everything is `float64` by default -- correct and simple for
  gradient-checking (numerical precision matters there), but 2x the
  memory bandwidth of `float32` for no accuracy benefit once training
  is stable.
- Masking is enforced with a **dense** array the whole time. Training
  never gets computationally cheaper as sparsity increases in this
  implementation -- only Part 4's separate, inference-only
  `scipy.sparse` path does. This is an intentional scope decision (a
  from-scratch autodiff engine that natively supports sparse tensors
  and sparse backward rules is a much larger undertaking), but it is
  the honest answer to "where's the bottleneck": at high sparsity, the
  dense mask multiply is pure waste that a training-time sparse format
  would eliminate.

**What we'd optimize, in order of expected impact:**

1. Cache the mask `Tensor` per layer instead of rebuilding it every
   forward call; skip the multiply entirely if a layer's mask is still
   all-ones (common early in training, before pruning starts).
2. Switch to `float32` once past gradient-checking.
3. Fuse common two-op patterns (bias-add + activation; the
   mask-multiply + matmul) into single ops with hand-written combined
   backward rules, cutting one allocation and one array pass per fusion.
4. For a static architecture (ours doesn't change shape between
   batches), trace the graph once and replay the same op sequence on
   new data rather than reconstructing Python objects every step --
   this is the idea behind `jax.jit`/graph-mode frameworks, and would
   be the highest-leverage change if this needed to scale past a toy
   model.
5. Once sparsity is high enough that it matters for training wall-clock
   (not just inference), move the masked weight matrices to an actual
   sparse storage format for the forward/backward matmul, not just at
   inference time in Part 4 -- this requires deriving sparse-aware
   backward rules (gradient of a sparse matmul is well-defined but
   needs its own implementation, not just NumPy's dense backward
   reused on a sparse array).

## Q4. How would you serve a self-pruned model in a real multi-tenant inference service at scale?

**Don't ship this engine to production.** `engine/` is a from-scratch,
pure-Python/NumPy research/training implementation; it exists to make
the pruning logic auditable and correct, not fast. Serving means
exporting the final (sparse) weights into a real inference runtime
(ONNX Runtime, TensorRT, a custom CUDA/Triton kernel, etc.) and never
touching this codebase's `Tensor`/`backward()` machinery again at
request time.

**Unstructured sparsity usually doesn't translate to hardware
speedup on its own** -- our own Part 4 measurement makes this
concrete: naive `scipy.sparse` CSR only starts beating dense NumPy
matmul above ~98-99% sparsity on a 512x512 layer; below that, per-
nonzero indexing overhead loses to BLAS's cache-friendly dense
kernels. The same story holds on GPUs, where unstructured sparsity
usually needs to reach the high-90s% before a generic sparse kernel
beats a dense one, and even then a naive sparse kernel is often memory-
bandwidth-bound rather than compute-bound. Two practical responses:

- Prefer **structured** sparsity for anything latency-sensitive:
  prune whole neurons/channels (structured pruning) so the "sparse"
  network is just a smaller *dense* network -- gets the speedup for
  free from ordinary dense kernels, no custom sparse kernel needed. Or
  target hardware-supported structured sparsity (e.g. NVIDIA's 2:4
  structured sparse tensor cores), which guarantees the speedup the
  hardware was built for instead of leaving it to a generic sparse
  format's mercy.
- If unstructured sparsity is required (e.g. because our saliency
  criterion's gains specifically come from irregular per-connection
  choices, as here), pair it with a serving-side compressed format
  (compact index lists per output neuron, block-sparse tiling aligned
  to the hardware's vector width) and *combine* with quantization
  (int8 weights) -- both reduce memory bandwidth, which is usually the
  real bottleneck for small-batch multi-tenant inference, more
  reliably than either alone reduces FLOPs.

**Multi-tenant specifics:**

- *Memory footprint is the actual lever, not FLOPs, for many serving
  workloads.* A model that's 4x smaller (90%+ prunable with a real
  compressed format) means 4x more tenant models resident in GPU
  memory/host RAM at once, which is usually what determines how many
  distinct fine-tunes/customers you can serve per accelerator before
  needing to page models in and out -- exactly AQUA's stated cost
  obsession.
- *Batching.* Because the sparsity pattern is fixed per model (not
  per-request), every request routed to the same tenant's model shares
  the same mask -- so continuous/dynamic batching across concurrent
  requests to that model works exactly like it would for a dense model
  of the reduced size; sparsity doesn't fight batching the way, say,
  per-example early-exit would.
- *Routing/isolation.* Different tenants likely have differently-
  pruned models (different sparsity targets, possibly different
  architectures). The serving layer needs per-model kernel/format
  metadata (structured vs. unstructured, achieved sparsity, quantized
  or not) so the request router can dispatch to the right compiled
  kernel rather than assuming one kernel fits all tenants.
- *Safe rollout.* Aggregate accuracy at a chosen sparsity (our Pareto
  curve) can hide long-tail regressions on rare inputs/classes that
  matter to a specific tenant even when overall accuracy looks fine --
  canary the pruned model against the dense one on live (or held-out,
  tenant-specific) traffic before fully cutting over, and keep the
  achieved-sparsity/accuracy pair logged per model version so a
  regression is traceable to a specific pruning run and seed.
