"""Threshold selection strategies and threshold-free metrics.

An anomaly score is only half the story; a deployment needs a threshold.
Three strategies are compared, all tuned on VALIDATION data only:

1. precision@k — flag the k highest scores (k = expected fraud count);
   reports precision among the top-k. Operational reading: "if the review
   team can inspect k transactions, how many are fraud?"
2. f1-optimal — threshold maximizing F1 on validation.
3. fixed-FPR — threshold giving a target false-positive rate on validation
   normals. Operational reading: "hold the false-alarm rate at x%."
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores))


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (area under the PR curve) — the honest metric here."""
    return float(average_precision_score(y_true, scores))


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of frauds among the k highest-scoring transactions."""
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, len(scores))
    topk = np.argsort(scores)[-k:]
    return float(y_true[topk].mean())


def threshold_for_fpr(
    y_true: np.ndarray, scores: np.ndarray, target_fpr: float
) -> float:
    """Threshold with FPR on negatives approximately <= target_fpr.

    Uses the 'higher' quantile (an observed score value), so the achieved FPR
    is conservative up to the 1/n resolution of the empirical quantile.
    """
    neg = scores[y_true == 0]
    return float(np.quantile(neg, 1.0 - target_fpr, method="higher"))


def threshold_for_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Threshold maximizing F1; returns (threshold, best_f1)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    # precision_recall_curve returns len(thresholds) == len(precision) - 1
    best_idx = int(np.argmax(f1))
    best_thr = float(thresholds[min(best_idx, len(thresholds) - 1)])
    return best_thr, float(f1[best_idx])


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(np.int64)


def confusion_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, int | float]:
    pred = apply_threshold(scores, threshold)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "f1": f1_score(y_true, pred, zero_division=0),
    }
