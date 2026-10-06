"""Tests for scoring metrics, threshold strategies, and evaluation helpers."""

import numpy as np
import pytest

from anomaly import evaluation as E
from anomaly import scoring as S


@pytest.fixture()
def toy():
    rng = np.random.default_rng(11)
    y = np.array([0] * 90 + [1] * 10)
    # informative scores: frauds mostly high
    s = rng.normal(size=100)
    s[90:] += 2.5
    return y, s


def test_roc_auc_and_pr_auc_range(toy):
    y, s = toy
    assert 0.5 < S.roc_auc(y, s) <= 1.0
    assert 0.0 < S.pr_auc(y, s) <= 1.0
    # PR-AUC must beat the base rate for an informative score
    assert S.pr_auc(y, s) > y.mean()


def test_precision_at_k(toy):
    y, s = toy
    p10 = S.precision_at_k(y, s, 10)
    assert 0.0 <= p10 <= 1.0
    # perfect score vector => precision@10 == 1
    perfect = np.where(y == 1, 1.0, 0.0)
    assert S.precision_at_k(y, perfect, 10) == 1.0
    with pytest.raises(ValueError):
        S.precision_at_k(y, s, 0)


def test_threshold_for_fpr_hits_target(toy):
    y, s = toy
    thr = S.threshold_for_fpr(y, s, 0.05)
    fpr = (s[y == 0] >= thr).mean()
    # Quantile-based FPR control is exact up to the 1/n empirical resolution
    # (here n_neg=90, so allow ~1 extra false positive of slack).
    assert fpr <= 0.05 + 2 / (y == 0).sum()


def test_threshold_for_f1_reasonable(toy):
    y, s = toy
    thr, f1 = S.threshold_for_f1(y, s)
    assert 0.0 <= f1 <= 1.0
    assert s.min() <= thr <= s.max()


def test_confusion_at_threshold_sums(toy):
    y, s = toy
    c = S.confusion_at_threshold(y, s, thr := float(np.median(s)))
    assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == len(y)
    assert 0.0 <= c["precision"] <= 1.0
    assert 0.0 <= c["recall"] <= 1.0


def test_evaluate_method_and_table(toy):
    y, s = toy
    r1 = E.evaluate_method("a", y, s)
    r2 = E.evaluate_method("b", y, 1 - s)  # inverted: worse
    table = E.comparison_table([r1, r2])
    assert list(table["method"]) == ["a", "b"]  # sorted by pr_auc desc
    assert "precision@100" in table.columns


def test_score_separation(toy):
    y, s = toy
    d = E.score_separation(y, s)
    assert d["n_fraud"] == 10 and d["n_normal"] == 90
    assert d["median_fraud"] > d["median_normal"]
    assert 0.0 <= d["frac_fraud_above_p99_normal"] <= 1.0
