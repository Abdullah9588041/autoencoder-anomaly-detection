"""Autoencoder training loop with early stopping on normal validation loss."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import Autoencoder


def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_autoencoder(
    model: Autoencoder,
    train_loader: DataLoader,
    X_val_normal: np.ndarray,
    epochs: int = 60,
    patience: int = 6,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """Train on normal transactions only; early-stop on normal val MSE.

    Returns a history dict with per-epoch train/val loss and training metadata.
    """
    seed_everything(seed)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    Xv = torch.from_numpy(X_val_normal.astype(np.float32))

    best_val = float("inf")
    best_state: dict | None = None
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), Xv).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    history["epochs_run"] = epoch + 1
    history["best_val_loss"] = best_val
    return history
