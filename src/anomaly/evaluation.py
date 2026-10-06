"""Method comparison and score-distribution diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .scoring import pr_auc, precision_at_k, roc_auc


def evaluate_method(
    name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    k_values: tuple[int, ...] = (100, 500),
) -> dict:
    """Threshold-free metrics + precision@k for one method."""
    out: dict = {
        "method": name,
        "roc_auc": roc_auc(y_true, scores),
        "pr_auc": pr_auc(y_true, scores),
    }
    for k in k_values:
        out[f"precision@{k}"] = precision_at_k(y_true, scores, k)
    out["mean_score_fraud"] = float(scores[y_true == 1].mean())
    out["mean_score_normal"] = float(scores[y_true == 0].mean())
    return out


def comparison_table(results: list[dict]) -> pd.DataFrame:
    """One row per method; sorted by PR-AUC (the metric that matters here)."""
    df = pd.DataFrame(results)
    return df.sort_values("pr_auc", ascending=False).reset_index(drop=True)


def score_separation(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """How well do fraud scores separate from normal scores?"""
    s_fraud = scores[y_true == 1]
    s_normal = scores[y_true == 0]
    return {
        "n_fraud": int(len(s_fraud)),
        "n_normal": int(len(s_normal)),
        "median_fraud": float(np.median(s_fraud)),
        "median_normal": float(np.median(s_normal)),
        "p90_normal": float(np.quantile(s_normal, 0.90)),
        "p99_normal": float(np.quantile(s_normal, 0.99)),
        "frac_fraud_above_p99_normal": float((s_fraud > np.quantile(s_normal, 0.99)).mean()),
    }
