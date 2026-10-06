"""Tests for data loading, splitting, and scaling protocol."""

import numpy as np
import pandas as pd
import pytest

from anomaly import data as D


@pytest.fixture()
def tiny_csv(tmp_path):
    """Small synthetic fraud-like CSV with the expected column layout."""
    rng = np.random.default_rng(0)
    n, p = 2000, 28
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    df = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
    df["Class"] = (rng.random(n) < 0.05).astype(int)
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)
    return path


def test_load_dataframe_rejects_wrong_shape(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="Unexpected shape"):
        D.load_dataframe(bad)


def test_prepare_data_splits_and_scales(tiny_csv, monkeypatch):
    # Patch the strict ULB verification so the tiny fixture loads.
    monkeypatch.setattr(D, "EXPECTED_ROWS", 2000)
    monkeypatch.setattr(D, "EXPECTED_FRAUDS", int(pd.read_csv(tiny_csv)["Class"].sum()))
    fd = D.prepare_data(tiny_csv, seed=1)
    n_total = len(fd.y_train) + len(fd.y_val) + len(fd.y_test)
    assert n_total == 2000
    # Stratification preserved the fraud rate approximately in each split.
    for y in (fd.y_train, fd.y_val, fd.y_test):
        assert abs(y.mean() - 0.05) < 0.02
    # No frauds in the normal-only training subset (no label leakage).
    assert (fd.y_train == 1).sum() > 0  # fixture does contain frauds overall
    assert fd.X_train_normal.shape[0] == (fd.y_train == 0).sum()
    # Scaler was fit on normals only: transformed normal-train mean ~ 0.
    assert abs(fd.X_train_normal.mean()) < 0.05
    assert fd.X_train.shape[1] == 29


def test_scaler_fit_on_normals_only(tiny_csv, monkeypatch):
    """The scaler's mean must equal the normal-subset mean, not the full mean."""
    monkeypatch.setattr(D, "EXPECTED_ROWS", 2000)
    monkeypatch.setattr(D, "EXPECTED_FRAUDS", int(pd.read_csv(tiny_csv)["Class"].sum()))
    df = pd.read_csv(tiny_csv)
    X = df[D.FEATURE_COLS].to_numpy()
    y = df[D.TARGET_COL].to_numpy(dtype=np.int64)
    # Replicate prepare_data's exact two-stage stratified split (seed=1).
    from sklearn.model_selection import train_test_split

    X_tmp, _, y_tmp, _ = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )
    X_train, _, y_train, _ = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=1, stratify=y_tmp
    )
    fd = D.prepare_data(tiny_csv, seed=1)
    normal_mean = X_train[y_train == 0].mean(axis=0)
    np.testing.assert_allclose(fd.scaler.mean_, normal_mean, rtol=1e-8)


def test_normal_loader_batches():
    X = np.random.default_rng(2).normal(size=(100, 29))
    loader = D.normal_loader(X, batch_size=32, seed=3)
    batches = list(loader)
    assert len(batches) == 4  # 32+32+32+4
    assert batches[0][0].shape == (32, 29)
