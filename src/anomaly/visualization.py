"""Matplotlib figures for the anomaly-detection analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def score_histograms(
    y_true: np.ndarray, scores_by_method: dict[str, np.ndarray], path: str | Path
) -> None:
    n = len(scores_by_method)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (name, scores) in zip(axes, scores_by_method.items()):
        ax.hist(scores[y_true == 0], bins=80, alpha=0.6, label="normal", density=True)
        ax.hist(scores[y_true == 1], bins=40, alpha=0.8, label="fraud", density=True)
        ax.set_title(f"{name}: anomaly-score distributions")
        ax.set_xlabel("anomaly score")
        ax.legend()
    fig.tight_layout()
    _save(fig, path)


def pr_curves(
    y_true: np.ndarray, scores_by_method: dict[str, np.ndarray], path: str | Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    base_rate = y_true.mean()
    for name, scores in scores_by_method.items():
        p, r, _ = precision_recall_curve(y_true, scores)
        ax.plot(r, p, label=name)
    ax.axhline(base_rate, color="k", linestyle="--", label=f"baseline ({base_rate:.4f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision–Recall curves (test set)")
    ax.legend()
    _save(fig, path)


def roc_curves(
    y_true: np.ndarray, scores_by_method: dict[str, np.ndarray], path: str | Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, scores in scores_by_method.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves (test set) — note the deceptively high values")
    ax.legend()
    _save(fig, path)


def latent_projection(
    latent: np.ndarray, y_true: np.ndarray, path: str | Path, seed: int = 42
) -> None:
    """2-D PCA projection of autoencoder bottleneck vectors."""
    rng = np.random.default_rng(seed)
    normal_idx = np.where(y_true == 0)[0]
    fraud_idx = np.where(y_true == 1)[0]
    # subsample normals for a readable plot
    keep = rng.choice(normal_idx, size=min(5_000, len(normal_idx)), replace=False)
    idx = np.concatenate([keep, fraud_idx])
    Z = PCA(n_components=2, random_state=seed).fit_transform(latent[idx])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Z[: len(keep), 0], Z[: len(keep), 1], s=4, alpha=0.3, label="normal")
    ax.scatter(Z[len(keep) :, 0], Z[len(keep) :, 1], s=12, alpha=0.9, label="fraud")
    ax.set_title("Autoencoder latent space (PCA-2D, test set)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    _save(fig, path)


def threshold_tradeoff(
    y_true: np.ndarray, scores: np.ndarray, path: str | Path, n_points: int = 200
) -> None:
    """Precision / recall / F1 / FPR as functions of the threshold."""
    thresholds = np.quantile(scores, np.linspace(0.5, 1.0, n_points))
    precisions, recalls, f1s, fprs = [], [], [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        precisions.append(tp / (tp + fp) if tp + fp else 0.0)
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        f1s.append(
            2 * precisions[-1] * recalls[-1] / (precisions[-1] + recalls[-1] + 1e-12)
        )
        fprs.append(fp / (fp + tn) if fp + tn else 0.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, label="precision")
    ax.plot(thresholds, recalls, label="recall")
    ax.plot(thresholds, f1s, label="F1")
    ax.set_xlabel("anomaly-score threshold")
    ax.set_ylabel("metric")
    ax.set_title("Threshold trade-off (test set, autoencoder)")
    ax.legend()
    _save(fig, path)
