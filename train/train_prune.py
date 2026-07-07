"""
Part 3: train the MLP so it prunes itself down to a target sparsity
under a cubic schedule, while tracking accuracy throughout.

Run:
    python -m train.train_prune --final-sparsity 0.9 --criterion saliency

Saves results/part3_prune_curve_<criterion>_<sparsity>.{csv,png} showing
sparsity and test accuracy over the course of training.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from engine.tensor import Tensor
from optim.adam import Adam
from prune.pruner import Pruner
from train.common import build_model, evaluate
from train.data import iterate_minibatches, load_digits_split

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def train_with_pruning(seed: int = 0, epochs: int = 60, batch_size: int = 32, lr: float = 1e-3,
                        criterion: str = "saliency", final_sparsity: float = 0.9,
                        prune_start_frac: float = 0.05, prune_end_frac: float = 0.6,
                        prune_freq: int = 20, allow_regrowth: bool = False,
                        regrowth_fraction: float = 0.0, verbose: bool = True):
    X_train, y_train, X_test, y_test = load_digits_split(seed=seed)
    model = build_model(seed=seed)
    opt = Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 1000)

    steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
    total_steps = steps_per_epoch * epochs
    start_step = int(prune_start_frac * total_steps)
    end_step = int(prune_end_frac * total_steps)

    pruner = Pruner(model, opt, criterion=criterion, final_sparsity=final_sparsity,
                     start_step=start_step, end_step=end_step, prune_freq=prune_freq,
                     allow_regrowth=allow_regrowth, regrowth_fraction=regrowth_fraction)

    history = []
    step = 0
    for epoch in range(1, epochs + 1):
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size, rng):
            opt.zero_grad()
            logits = model(Tensor(xb))
            loss = logits.softmax_cross_entropy(yb)
            loss.backward()
            assert np.isfinite(loss.data), "loss went non-finite (NaN/Inf) during pruning"
            pruner.update_ema()
            opt.step()
            model.apply_masks()  # defense in depth: hard re-zero after the optimizer step
            pruner.maybe_prune(step)
            if allow_regrowth:
                pruner.maybe_regrow(step, xb, yb)

            epoch_loss += float(loss.data)
            n_batches += 1
            step += 1

        stats = model.sparsity_stats()
        test_loss, test_acc = evaluate(model, X_test, y_test)
        history.append({"epoch": epoch, "step": step, "train_loss": epoch_loss / n_batches,
                         "test_acc": test_acc, "sparsity": stats["sparsity"]})
        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            print(f"epoch {epoch:3d}  step {step:5d}  loss={epoch_loss / n_batches:.4f}  "
                  f"test_acc={test_acc:.4f}  sparsity={stats['sparsity']:.4f}")

    final_stats = model.sparsity_stats()
    final_test_loss, final_test_acc = evaluate(model, X_test, y_test)
    return model, history, pruner, {"final_sparsity": final_stats["sparsity"], "final_test_acc": final_test_acc}


def save_history(history, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_history(history, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(epochs, [h["test_acc"] for h in history], color="tab:blue", label="test accuracy")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("test accuracy", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["sparsity"] for h in history], color="tab:red", label="sparsity")
    ax2.set_ylabel("sparsity", color="tab:red")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--criterion", choices=["saliency", "magnitude"], default="saliency")
    parser.add_argument("--final-sparsity", type=float, default=0.9)
    parser.add_argument("--allow-regrowth", action="store_true")
    parser.add_argument("--regrowth-fraction", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, history, pruner, summary = train_with_pruning(
        seed=args.seed, epochs=args.epochs, criterion=args.criterion,
        final_sparsity=args.final_sparsity, allow_regrowth=args.allow_regrowth,
        regrowth_fraction=args.regrowth_fraction,
    )

    tag = f"{args.criterion}_{int(args.final_sparsity * 100)}"
    save_history(history, os.path.join(RESULTS_DIR, f"part3_prune_curve_{tag}.csv"))
    plot_history(history, os.path.join(RESULTS_DIR, f"part3_prune_curve_{tag}.png"),
                 f"Self-pruning ({args.criterion}, target sparsity={args.final_sparsity})")

    print(f"\nFinal sparsity: {summary['final_sparsity']:.4f}  "
          f"Final test accuracy: {summary['final_test_acc']:.4f}")
    print(f"Saved results/part3_prune_curve_{tag}.{{csv,png}}")
