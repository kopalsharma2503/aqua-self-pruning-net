import numpy as np

from engine.tensor import Tensor
from nn.layers import MLP
from optim.adam import Adam
from prune.pruner import Pruner
from prune.scheduler import cubic_sparsity


def test_cubic_schedule_endpoints_and_monotonic():
    start, end, final = 10, 110, 0.9
    assert cubic_sparsity(0, start, end, final) == 0.0
    assert cubic_sparsity(start, start, end, final) == 0.0
    assert cubic_sparsity(end, start, end, final) == final
    assert cubic_sparsity(end + 50, start, end, final) == final

    values = [cubic_sparsity(t, start, end, final) for t in range(start, end + 1, 5)]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:])), "schedule must be non-decreasing"


def _toy_model_and_data(seed=0):
    rng = np.random.default_rng(seed)
    model = MLP([8, 16, 4], rng)
    X = rng.normal(size=(20, 8))
    y = rng.integers(0, 4, size=20)
    return model, X, y


def test_pruner_hits_target_sparsity_and_is_monotonic_without_regrowth():
    model, X, y = _toy_model_and_data()
    opt = Adam(model.parameters(), lr=1e-2)
    pruner = Pruner(model, opt, criterion="saliency", final_sparsity=0.8,
                     start_step=0, end_step=40, prune_freq=5)

    prev_sparsity = 0.0
    for step in range(60):
        opt.zero_grad()
        logits = model(Tensor(X))
        loss = logits.softmax_cross_entropy(y)
        loss.backward()
        pruner.update_ema()
        opt.step()
        model.apply_masks()
        pruner.maybe_prune(step)

        current = model.sparsity_stats()["sparsity"]
        assert current >= prev_sparsity - 1e-9, "sparsity decreased without regrowth enabled"
        prev_sparsity = current

    assert abs(model.sparsity_stats()["sparsity"] - 0.8) < 0.02


def test_masked_weights_are_exact_zero_throughout_pruning():
    model, X, y = _toy_model_and_data()
    opt = Adam(model.parameters(), lr=1e-2)
    pruner = Pruner(model, opt, criterion="saliency", final_sparsity=0.9,
                     start_step=0, end_step=30, prune_freq=3)

    for step in range(40):
        opt.zero_grad()
        logits = model(Tensor(X))
        loss = logits.softmax_cross_entropy(y)
        loss.backward()
        pruner.update_ema()
        opt.step()
        model.apply_masks()
        pruner.maybe_prune(step)

    for layer in model.weight_layers():
        assert np.all(layer.W.data[layer.mask == 0] == 0.0)


def test_regrowth_can_revive_a_pruned_connection():
    model, X, y = _toy_model_and_data(seed=3)
    opt = Adam(model.parameters(), lr=1e-2)
    pruner = Pruner(model, opt, criterion="saliency", final_sparsity=0.7,
                     start_step=0, end_step=30, prune_freq=3,
                     allow_regrowth=True, regrowth_fraction=0.2)

    masks_over_time = []
    for step in range(50):
        opt.zero_grad()
        logits = model(Tensor(X))
        loss = logits.softmax_cross_entropy(y)
        loss.backward()
        pruner.update_ema()
        opt.step()
        model.apply_masks()
        pruner.maybe_prune(step)
        pruner.maybe_regrow(step, X, y)
        masks_over_time.append([l.mask.copy() for l in model.weight_layers()])

    # A connection that was pruned at some point and alive again later
    # would show a 0 -> 1 transition somewhere in its per-layer history.
    revived_any = False
    n_layers = len(masks_over_time[0])
    for li in range(n_layers):
        stacked = np.stack([snapshot[li] for snapshot in masks_over_time])  # (T, in, out)
        T = stacked.shape[0]
        flat = stacked.reshape(T, -1)
        for j in range(flat.shape[1]):
            seq = flat[:, j]
            if np.any((seq[:-1] == 0) & (seq[1:] == 1)):
                revived_any = True
                break
        if revived_any:
            break

    regrow_events = [h for h in pruner.history if h.get("event") == "regrow" and h.get("n_changed", 0) > 0]
    assert revived_any or regrow_events, "expected at least one connection to be revived under regrowth"
