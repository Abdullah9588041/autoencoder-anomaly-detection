"""End-to-end analysis: train all methods, tune thresholds on validation,
evaluate on test, save figures + metrics.

Usage:
    python scripts/run_analysis.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anomaly import data as D
from anomaly import evaluation as E
from anomaly import models as M
from anomaly import scoring as S
from anomaly import train as T
from anomaly import visualization as V

SEED = 42
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
DATA_PATH = ROOT / "data" / "creditcard.csv"
FIXED_FPR = 0.005  # 0.5% false-alarm rate operating point


def main() -> None:
    t0 = time.time()
    print("Loading data ...")
    fd = D.prepare_data(DATA_PATH, seed=SEED)
    print(f"  train: {fd.X_train.shape}, val: {fd.X_val.shape}, test: {fd.X_test.shape}")
    print(f"  train fraud rate: {fd.fraud_rate:.5f}")

    # ---------------- Autoencoder ----------------
    print("Training autoencoder ...")
    torch_device_note = "cpu"
    model = M.Autoencoder(input_dim=fd.X_train.shape[1])
    train_loader = D.normal_loader(fd.X_train_normal, seed=SEED)
    X_val_normal = fd.X_val[fd.y_val == 0]
    hist = T.train_autoencoder(model, train_loader, X_val_normal, seed=SEED)
    print(f"  epochs: {hist['epochs_run']}, best val MSE: {hist['best_val_loss']:.6f}")
    ae_val = M.reconstruction_errors(model, fd.X_val)
    ae_test = M.reconstruction_errors(model, fd.X_test)

    # ---------------- Baselines ----------------
    print("Fitting Isolation Forest ...")
    iforest = M.fit_isolation_forest(
        fd.X_train_normal, contamination=fd.fraud_rate, seed=SEED
    )
    if_val = M.isolation_forest_scores(iforest, fd.X_val)
    if_test = M.isolation_forest_scores(iforest, fd.X_test)

    print("Fitting One-Class SVM (10k subsample) ...")
    ocsvm = M.fit_one_class_svm(fd.X_train_normal, seed=SEED)
    svm_val = M.one_class_svm_scores(ocsvm, fd.X_val)
    svm_test = M.one_class_svm_scores(ocsvm, fd.X_test)

    print("Fitting supervised XGBoost reference ...")
    xgb = M.fit_xgboost_reference(fd.X_train, fd.y_train, seed=SEED)
    xgb_test = xgb.predict_proba(fd.X_test)[:, 1]
    xgb_val = xgb.predict_proba(fd.X_val)[:, 1]

    scores_test = {
        "autoencoder": ae_test,
        "isolation_forest": if_test,
        "one_class_svm": svm_test,
        "xgboost_supervised": xgb_test,
    }
    scores_val = {
        "autoencoder": ae_val,
        "isolation_forest": if_val,
        "one_class_svm": svm_val,
        "xgboost_supervised": xgb_val,
    }

    # ---------------- Threshold-free comparison ----------------
    results = [
        E.evaluate_method(name, fd.y_test, s) for name, s in scores_test.items()
    ]
    table = E.comparison_table(results)
    print("\n" + table.to_string(index=False))

    # ---------------- Threshold strategies (tuned on VALIDATION) ----------------
    n_fraud_val = int(fd.y_val.sum())
    strategies: dict[str, dict] = {}
    for name in ("autoencoder", "isolation_forest", "one_class_svm"):
        sv, st = scores_val[name], scores_test[name]
        k = n_fraud_val
        thr_k = float(np.sort(sv)[-k:][0])
        thr_f1, f1_val = S.threshold_for_f1(fd.y_val, sv)
        thr_fpr = S.threshold_for_fpr(fd.y_val, sv, FIXED_FPR)
        strategies[name] = {
            "precision@k": {
                "k": k,
                "threshold": thr_k,
                "test": S.confusion_at_threshold(fd.y_test, st, thr_k),
            },
            "f1_optimal": {
                "threshold": thr_f1,
                "val_f1": f1_val,
                "test": S.confusion_at_threshold(fd.y_test, st, thr_f1),
            },
            f"fixed_fpr_{FIXED_FPR}": {
                "threshold": thr_fpr,
                "test": S.confusion_at_threshold(fd.y_test, st, thr_fpr),
            },
        }

    # Chosen operating point: fixed FPR on the autoencoder.
    chosen = strategies["autoencoder"][f"fixed_fpr_{FIXED_FPR}"]
    print("\nChosen operating point (autoencoder, fixed FPR=0.5%):")
    print(" ", json.dumps(chosen["test"], indent=2))

    # ---------------- Figures ----------------
    print("Saving figures ...")
    V.score_histograms(fd.y_test, scores_test, FIGS / "score_histograms.png")
    V.pr_curves(fd.y_test, scores_test, FIGS / "pr_curves.png")
    V.roc_curves(fd.y_test, scores_test, FIGS / "roc_curves.png")
    V.threshold_tradeoff(fd.y_test, ae_test, FIGS / "threshold_tradeoff.png")
    with torch.no_grad():
        latent = model.encode(torch.from_numpy(fd.X_test.astype(np.float32))).numpy()
    V.latent_projection(latent, fd.y_test, FIGS / "latent_projection.png")

    # ---------------- Metrics record ----------------
    metrics = {
        "seed": SEED,
        "device": torch_device_note,
        "data": {
            "n_train": len(fd.y_train),
            "n_val": len(fd.y_val),
            "n_test": len(fd.y_test),
            "fraud_rate_train": fd.fraud_rate,
            "fraud_rate_test": float(fd.y_test.mean()),
        },
        "autoencoder": {
            "architecture": "29-16-8-4-8-16-29",
            "epochs_run": hist["epochs_run"],
            "best_val_mse": hist["best_val_loss"],
        },
        "comparison_test": table.to_dict(orient="records"),
        "threshold_strategies_validation_tuned": strategies,
        "chosen_operating_point": {
            "method": "autoencoder",
            "strategy": f"fixed_fpr_{FIXED_FPR}",
            "justification": (
                "A fixed false-alarm rate maps directly to review-team capacity: "
                f"at FPR={FIXED_FPR}, expected daily reviews = {FIXED_FPR} * daily volume. "
                "F1-optimal thresholds overfit the validation fraud mix; precision@k "
                "assumes a known fraud count, which is unknowable in production."
            ),
            **chosen,
        },
        "wall_clock_seconds": round(time.time() - t0, 1),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nDone in {metrics['wall_clock_seconds']}s. Metrics -> results/metrics.json")


if __name__ == "__main__":
    main()
