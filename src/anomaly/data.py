"""Data loading, splitting, and scaling for the ULB credit-card fraud dataset.

Protocol notes (leakage discipline):
- Splits are stratified so the ~0.17% fraud rate is preserved in every split.
- The StandardScaler is fit on NORMAL transactions of the training split only.
  Rationale: the scaler defines "what normal looks like"; fitting it on frauds
  would let anomalies shift the scale and partially mask themselves.
- The autoencoder (and all unsupervised baselines) are trained on the
  normal-only subset of the training split. Labels are used only for
  threshold tuning (validation) and final evaluation (test), plus the
  supervised XGBoost reference model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import torch

EXPECTED_ROWS = 284_807
EXPECTED_COLS = 31
EXPECTED_FRAUDS = 492
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COL = "Class"
SEED = 42


@dataclass
class FraudData:
    """Prepared dataset for anomaly-detection experiments."""

    X_train: np.ndarray          # full training features (for supervised reference)
    y_train: np.ndarray
    X_train_normal: np.ndarray   # normal-only training features (unsupervised training)
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    feature_names: list[str]

    @property
    def fraud_rate(self) -> float:
        return float(self.y_train.mean())


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and verify it matches the known ULB dataset."""
    df = pd.read_csv(path)
    n_rows, n_cols = df.shape
    if (n_rows, n_cols) != (EXPECTED_ROWS, EXPECTED_COLS):
        raise ValueError(
            f"Unexpected shape {(n_rows, n_cols)}; "
            f"expected {(EXPECTED_ROWS, EXPECTED_COLS)}. Wrong file?"
        )
    n_fraud = int(df[TARGET_COL].sum())
    if n_fraud != EXPECTED_FRAUDS:
        raise ValueError(
            f"Unexpected fraud count {n_fraud}; expected {EXPECTED_FRAUDS}."
        )
    return df


def prepare_data(
    path: str | Path,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = SEED,
) -> FraudData:
    """Stratified split + normal-only scaling. Returns a FraudData bundle."""
    df = load_dataframe(path)
    X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y = df[TARGET_COL].to_numpy(dtype=np.int64)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, random_state=seed, stratify=y_tmp
    )

    # Fit the scaler on normal training transactions only.
    scaler = StandardScaler()
    X_train_normal = X_train[y_train == 0]
    scaler.fit(X_train_normal)

    return FraudData(
        X_train=scaler.transform(X_train),
        y_train=y_train,
        X_train_normal=scaler.transform(X_train_normal),
        X_val=scaler.transform(X_val),
        y_val=y_val,
        X_test=scaler.transform(X_test),
        y_test=y_test,
        scaler=scaler,
        feature_names=FEATURE_COLS,
    )


def normal_loader(
    X_normal: np.ndarray, batch_size: int = 512, seed: int = SEED
) -> DataLoader:
    """DataLoader over normal transactions for autoencoder training."""
    gen = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(X_normal.astype(np.float32)))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)
