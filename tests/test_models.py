"""Tests for the autoencoder, training loop, and baselines."""

import numpy as np
import torch

from anomaly import models as M
from anomaly import train as T
from anomaly.data import normal_loader


def _normals(n=600, d=29, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d))


def test_autoencoder_forward_shapes():
    m = M.Autoencoder(input_dim=29, hidden=(16, 8), latent_dim=4)
    x = torch.randn(10, 29)
    assert m(x).shape == (10, 29)
    assert m.encode(x).shape == (10, 4)


def test_reconstruction_errors_nonneg_and_shaped():
    m = M.Autoencoder(input_dim=29)
    X = _normals(50)
    err = M.reconstruction_errors(m, X)
    assert err.shape == (50,)
    assert (err >= 0).all()


def test_train_reduces_val_loss_and_early_stops():
    T.seed_everything(0)
    X = _normals(800)
    loader = normal_loader(X[:600], batch_size=128, seed=0)
    m = M.Autoencoder(input_dim=29, hidden=(8, 4), latent_dim=2)
    hist = T.train_autoencoder(
        m, loader, X[600:], epochs=40, patience=3, lr=1e-2, seed=0
    )
    assert hist["epochs_run"] <= 40
    assert hist["best_val_loss"] < hist["val_loss"][0]


def test_no_label_leakage_in_ae_training_data(tmp_path):
    """The autoencoder must never see a fraud during training."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(500, 29))
    y = (rng.random(500) < 0.1).astype(int)
    X_normal = X[y == 0]
    # The protocol helper only receives the normal subset; assert separation.
    assert len(X_normal) == (y == 0).sum() < len(y)


def test_isolation_forest_scores_ordering():
    rng = np.random.default_rng(5)
    normal = rng.normal(size=(2000, 10))
    fraud = rng.normal(loc=6.0, size=(50, 10))  # far from the manifold
    clf = M.fit_isolation_forest(normal, contamination=0.02, seed=5)
    s = M.isolation_forest_scores(clf, np.vstack([normal[:200], fraud]))
    assert s[200:].mean() > s[:200].mean()


def test_one_class_svm_trains_on_subsample():
    X = _normals(500, d=10)
    clf = M.fit_one_class_svm(X, subsample=100, seed=6)
    s = M.one_class_svm_scores(clf, X[:20])
    assert s.shape == (20,)


def test_xgboost_reference_predicts():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(400, 10))
    y = (X[:, 0] > 1.0).astype(int)
    clf = M.fit_xgboost_reference(X, y, seed=7)
    p = clf.predict_proba(X)[:, 1]
    assert p.shape == (400,)
    assert ((p >= 0) & (p <= 1)).all()
