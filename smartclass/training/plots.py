"""Report figures. Uses matplotlib's object API + Agg canvas (no pyplot), so it is safe in a QThread."""
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def _save(fig, path):
    FigureCanvasAgg(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_curves(history, path, title, best=None):
    """history: list of dicts with epoch, train_loss, val_loss, train_acc, val_acc."""
    ep = [h["epoch"] for h in history]
    fig = Figure(figsize=(10, 4))
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(ep, [h["train_loss"] for h in history], label="train")
    ax1.plot(ep, [h["val_loss"] for h in history], label="validation")
    ax1.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax2.plot(ep, [h["train_acc"] for h in history], label="train")
    ax2.plot(ep, [h["val_acc"] for h in history], label="validation")
    ax2.set(xlabel="epoch", ylabel="accuracy", title="Accuracy", ylim=(0, 1.02))
    if best is None:
        best = max(history, key=lambda h: h["val_f1"])["epoch"]
    for ax in (ax1, ax2):
        ax.axvline(best, color="grey", ls="--", lw=1, label=f"best (epoch {best})")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    _save(fig, path)


def plot_confusion(cm, classes, path, title):
    """Row-normalised confusion matrix (recall per class), with the raw counts written in the cells."""
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig = Figure(figsize=(7, 6))
    ax = fig.subplots()
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="predicted", ylabel="true", title=title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)


def plot_comparison(results, path):
    """Bar chart: test accuracy and macro-F1 of every model."""
    names = [r["model"] for r in results]
    x = np.arange(len(names))
    fig = Figure(figsize=(max(6, 1.6 * len(names)), 4))
    ax = fig.subplots()
    ax.bar(x - 0.2, [r["test_acc"] for r in results], 0.4, label="test accuracy")
    ax.bar(x + 0.2, [r["test_macro_f1"] for r in results], 0.4, label="test macro-F1")
    for i, r in enumerate(results):
        ax.text(i - 0.2, r["test_acc"] + 0.01, f"{r['test_acc']:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, r["test_macro_f1"] + 0.01, f"{r['test_macro_f1']:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, names)
    ax.set(ylim=(0, 1.1), ylabel="score", title="Gesture models on unseen people (test split)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    _save(fig, path)
