"""Models: deep autoencoder (PyTorch) + unsupervised and supervised baselines."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier


class Autoencoder(nn.Module):
    """Symmetric deep autoencoder.

    Architecture: input -> hidden[0] -> hidden[1] -> latent
                       -> hidden[1] -> hidden[0] -> input.
    ReLU activations; linear output (features are standardized).
    """

    def __init__(
        self, input_dim: int, hidden: tuple[int, int] = (16, 8), latent_dim: int = 4
    ) -> None:
        super().__init__()
        h1, h2 = hidden
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
        )
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def reconstruction_errors(model: Autoencoder, X: np.ndarray) -> np.ndarray:
    """Per-sample mean-squared reconstruction error (the anomaly score)."""
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X.astype(np.float32))
        recon = model(Xt)
        err = ((Xt - recon) ** 2).mean(dim=1).numpy()
    return err


def fit_isolation_forest(
    X_normal: np.ndarray, contamination: float, seed: int = 42
) -> IsolationForest:
    """Isolation Forest trained on normal transactions.

    `contamination` is the expected anomaly fraction; here it is set to the
    known overall fraud rate so the model's score threshold is calibrated to
    the problem (documented, not tuned on labels).
    """
    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_normal)
    return clf


def isolation_forest_scores(clf: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous (negated decision function)."""
    return -clf.decision_function(X)


def fit_one_class_svm(
    X_normal: np.ndarray, subsample: int = 10_000, seed: int = 42, nu: float = 0.02
) -> OneClassSVM:
    """One-Class SVM trained on a subsample of normals.

    The full normal training set (~170k rows) makes the kernel matrix
    prohibitive; a 10k stratified-random subsample keeps training tractable.
    This is a documented limitation, not a tuned choice.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_normal), size=min(subsample, len(X_normal)), replace=False)
    clf = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
    clf.fit(X_normal[idx])
    return clf


def one_class_svm_scores(clf: OneClassSVM, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous (negated decision function)."""
    return -clf.decision_function(X)


def fit_xgboost_reference(
    X_train: np.ndarray, y_train: np.ndarray, seed: int = 42
) -> XGBClassifier:
    """Supervised XGBoost trained WITH labels — the 'cheating' reference.

    Included to quantify what supervision buys over unsupervised methods.
    scale_pos_weight compensates the ~0.17% positive rate.
    """
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train)
    return clf
