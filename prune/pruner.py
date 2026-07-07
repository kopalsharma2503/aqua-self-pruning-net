"""
Orchestrates gradual, global, unstructured pruning during training.

"Global" means: at each pruning event we pool importance scores across
every prunable weight matrix in the network and cut a single global
percentile threshold, rather than pruning each layer to the same
per-layer sparsity. This lets the schedule take more capacity from
layers that turn out to have more redundancy and less from layers where
every connection matters, instead of assuming all layers are equally
prunable.

Masking is enforced through `Linear.set_mask`, which (a) hard-zeros the
newly-pruned weights immediately and (b) reports exactly which
positions flipped so we can reset that optimizer's per-element Adam
state at those positions -- see optim/adam.py and the Part 1
correctness requirement.
"""
from __future__ import annotations

import numpy as np

from engine.tensor import Tensor
from prune.criteria import magnitude_score, saliency_score
from prune.scheduler import cubic_sparsity


class Pruner:
    def __init__(self, model, optimizer, criterion: str = "saliency", final_sparsity: float = 0.9,
                 start_step: int = 0, end_step: int = 1000, prune_freq: int = 50,
                 ema_decay: float = 0.9, allow_regrowth: bool = False, regrowth_fraction: float = 0.0):
        assert criterion in ("saliency", "magnitude")
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.final_sparsity = final_sparsity
        self.start_step = start_step
        self.end_step = end_step
        self.prune_freq = prune_freq
        self.ema_decay = ema_decay
        self.allow_regrowth = allow_regrowth
        self.regrowth_fraction = regrowth_fraction

        self.ema_scores = {id(l): np.zeros_like(l.W.data) for l in model.weight_layers()}
        self.history = []

    # ------------------------------------------------------------------
    def update_ema(self):
        """Call once per training step, right after loss.backward() and
        before optimizer.step() zeros the batch's grad on the next
        iteration (grad is still valid at this point)."""
        if self.criterion != "saliency":
            return
        for layer in self.model.weight_layers():
            g = layer.W.grad
            if g is None:
                continue
            score = saliency_score(layer.W.data, g)
            key = id(layer)
            self.ema_scores[key] = self.ema_decay * self.ema_scores[key] + (1 - self.ema_decay) * score

    def current_scores(self, layer) -> np.ndarray:
        if self.criterion == "magnitude":
            return magnitude_score(layer.W.data)
        return self.ema_scores[id(layer)]

    def target_sparsity(self, step: int) -> float:
        return cubic_sparsity(step, self.start_step, self.end_step, self.final_sparsity)

    # ------------------------------------------------------------------
    def maybe_prune(self, step: int):
        if step < self.start_step or step > self.end_step:
            return None
        if (step - self.start_step) % self.prune_freq != 0:
            return None

        target = self.target_sparsity(step)
        layers = self.model.weight_layers()
        all_scores = np.concatenate([self.current_scores(l).ravel() for l in layers])
        n_total = all_scores.size
        n_prune = int(round(target * n_total))

        if n_prune <= 0:
            threshold = -np.inf
        elif n_prune >= n_total:
            threshold = np.inf
        else:
            threshold = np.partition(all_scores, n_prune - 1)[n_prune - 1]

        for layer in layers:
            scores = self.current_scores(layer)
            new_mask = (scores > threshold).astype(np.float64)
            changed = layer.set_mask(new_mask)
            self.optimizer.reset_state_at(layer.W, changed)

        achieved = self.model.sparsity_stats()["sparsity"]
        record = {"step": step, "target_sparsity": target, "achieved_sparsity": achieved,
                  "event": "prune"}
        self.history.append(record)
        return record

    # ------------------------------------------------------------------
    def maybe_regrow(self, step: int, x_batch: np.ndarray, y_batch: np.ndarray):
        """
        Bonus: allow a pruned connection to come back if it becomes
        important again.

        Direct saliency/magnitude scores can NEVER regrow a masked
        connection: a masked weight is exactly 0, so both |W| and
        |W * grad| are exactly 0 for it (the mask forces dL/dW = 0 too,
        see nn/layers.py). There is no signal to rank pruned candidates
        against each other under those criteria -- they're all tied at
        zero.

        To get a real regrowth signal we borrow the RigL (Evci et al.,
        2020) trick: periodically run one extra forward/backward pass
        with the mask *temporarily lifted* (a "dense probe"), purely to
        read off `dL/dW` at currently-pruned positions as if they were
        still connected. This probe is never used to update the real
        weights or take an optimizer step -- it only ranks regrowth
        candidates by |probe_grad|. The highest-probe-gradient pruned
        connections are revived (reinitialized to 0, since that's what
        they already are), and to hold the schedule's sparsity target we
        prune an equal number of the currently-weakest surviving
        connections in the same step.

        Stability note (see DESIGN.md for the fuller discussion): a
        revived weight starts at exactly 0 with fresh (m=0, v=0, t=0)
        optimizer state, so its first few updates behave like a
        completely new parameter, not a perturbation of whatever least-
        important weight it replaced. That reset is what keeps this from
        corrupting training -- without it, a naive implementation could
        carry over stale Adam moments from the connection's pre-prune
        life and destabilize the following steps.
        """
        if not self.allow_regrowth or self.regrowth_fraction <= 0:
            return None
        if step < self.start_step or step > self.end_step:
            return None
        if (step - self.start_step) % self.prune_freq != 0:
            return None

        layers = self.model.weight_layers()
        saved_masks = {id(l): l.mask.copy() for l in layers}
        for l in layers:
            l.mask = np.ones_like(l.mask)
        for p in self.model.parameters():
            p.zero_grad()
        logits = self.model(Tensor(x_batch))
        loss = logits.softmax_cross_entropy(y_batch)
        loss.backward()
        probe_grads = {id(l): l.W.grad.copy() for l in layers}
        for l in layers:
            l.mask = saved_masks[id(l)]

        total_changed = 0
        for layer in layers:
            key = id(layer)
            mask = layer.mask
            pruned_flat = (mask.ravel() == 0)
            n_pruned = int(pruned_flat.sum())
            n_regrow = int(round(self.regrowth_fraction * n_pruned))
            if n_regrow <= 0 or n_pruned == 0:
                continue

            probe_score = np.abs(probe_grads[key]).ravel()
            candidate_score = np.where(pruned_flat, probe_score, -np.inf)
            n_regrow = min(n_regrow, n_pruned)
            revive_idx = np.argpartition(candidate_score, -n_regrow)[-n_regrow:]

            new_mask_flat = mask.ravel().copy()
            new_mask_flat[revive_idx] = 1.0

            # Hold the sparsity budget: cut an equal number of the
            # currently-weakest surviving connections elsewhere in this
            # layer. The just-revived positions are exempt from being cut
            # in this same pass -- their EMA saliency score is still near
            # zero (it hasn't had a chance to update yet, since it was 0
            # every step they were pruned), so without this exemption they
            # would immediately look like the "weakest alive" connections
            # and get cut right back out, silently cancelling the regrowth.
            alive_flat = new_mask_flat == 1
            alive_scores = self.current_scores(layer).ravel()
            cut_candidates = np.where(alive_flat, alive_scores, np.inf)
            cut_candidates[revive_idx] = np.inf
            n_cut = min(n_regrow, int(alive_flat.sum()) - len(revive_idx))
            cut_idx = np.argpartition(cut_candidates, n_cut - 1)[:n_cut] if n_cut > 0 else np.array([], dtype=int)
            new_mask_flat[cut_idx] = 0.0

            new_mask = new_mask_flat.reshape(mask.shape)
            changed = layer.set_mask(new_mask)
            self.optimizer.reset_state_at(layer.W, changed)
            total_changed += int(changed.sum())

        if total_changed:
            self.history.append({"step": step, "event": "regrow", "n_changed": total_changed})
        return total_changed
