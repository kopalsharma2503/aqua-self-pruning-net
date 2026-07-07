"""
Part 2: train the MLP with our own Adam optimizer and mini-batch loop,
no pruning. Reports a per-epoch learning curve (loss + accuracy) and
saves it to results/part2_learning_curve.{csv,png}.

Run:
    python -m train.train_baseline
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from engine.tensor import Tensor
from optim.adam import Adam
from train.common import build_model, evaluate
from train.data import iterate_minibatches, load_digits_split

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def train(seed: int = 0, epochs: int = 60, batch_size: int = 32, lr: float = 1e-3, verbose: bool = True):
    X_train, y_train, X_test, y_test = load_digits_split(seed=seed)
    model = build_model(seed=seed)
    opt = Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 1000)

    history = []
    for epoch in range(1, epochs + 1):
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size, rng):
            opt.zero_grad()
            logits = model(Tensor(xb))
            loss = logits.softmax_cross_entropy(yb)
            loss.backward()
            opt.step()
            assert np.isfinite(loss.data), "loss went non-finite (NaN/Inf) during training"
            epoch_loss += float(loss.data)
            n_batches += 1

        train_loss = epoch_loss / n_batches
        test_loss, test_acc = evaluate(model, X_test, y_test)
        _, train_acc = evaluate(model, X_train, y_train)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "test_loss": test_loss, "test_acc": test_acc})
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                  f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

    for p in model.parameters():
        assert np.isfinite(p.data).all(), "NaN/Inf detected in final parameters"

    return model, history


def save_history(history, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["test_loss"] for h in history], label="test")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss"); axes[0].legend()
    axes[0].set_title("Loss")

    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs, [h["test_acc"] for h in history], label="test")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend()
    axes[1].set_title("Accuracy")

    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, history = train(seed=args.seed, epochs=args.epochs)
    save_history(history, os.path.join(RESULTS_DIR, "part2_learning_curve.csv"))
    plot_history(history, os.path.join(RESULTS_DIR, "part2_learning_curve.png"))
    print(f"\nFinal test accuracy: {history[-1]['test_acc']:.4f}")
    print("Saved results/part2_learning_curve.{csv,png}")
