"""Result figures: confusion matrix, ROC curve and training history."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402

from . import config  # noqa: E402


def plot_confusion_matrix(y_true, y_pred, out: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
                annot_kws={"size": 14}, ax=ax)
    ax.set(xlabel="Predicted label", ylabel="True label", title=title)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_roc(y_true, y_score, auc: float, out: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="grey", label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate",
           ylabel="True positive rate", title=title)
    ax.legend(loc="lower right")
    ax.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_history(history: dict, out: Path, title: str) -> None:
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.plot(epochs, history[key], "o-", ms=4, label="Train")
        ax.plot(epochs, history[f"val_{key}"], "o-", ms=4, label="Validation")
        ax.set(xlabel="Epoch", ylabel=label, title=label)
        ax.grid(ls="--", alpha=0.4)
        ax.legend()
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
