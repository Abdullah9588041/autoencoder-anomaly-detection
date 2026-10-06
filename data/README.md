# Data

## Source (real data)

`creditcard.csv` is the **ULB Credit Card Fraud Detection** dataset
(`mlg-ulb/creditcardfraud` on Kaggle): 284,807 anonymized European
card transactions from September 2013, of which 492 (0.173%) are frauds.
Features `V1`–`V28` are PCA components; `Time` is seconds elapsed since the
first transaction; `Amount` is the transaction amount; `Class` is the label
(1 = fraud).

Because the Kaggle API requires credentials, this project downloads the file
from a public GitHub mirror instead:

- **Mirror URL:** https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv
- **Accessed:** 2026-10-06
- **Verification** (`python data/download.py`): 284,807 rows × 31 columns,
  header `Time,V1..V28,Amount,Class`, fraud count exactly 492 — matching the
  published dataset statistics.

The original dataset is released for research use by the ULB Machine Learning
Group (see Dal Pozzolo et al. references in the main README). The raw CSV is
**git-ignored** (102 MB); only this documentation and the download script are
tracked. To reproduce: run `python data/download.py`.

## Columns

| Column | Meaning |
|---|---|
| `Time` | Seconds elapsed since first transaction (dropped as a feature; see below) |
| `V1`–`V28` | Anonymized PCA features (already standardized) |
| `Amount` | Transaction amount (standardized by our pipeline) |
| `Class` | 1 = fraud, 0 = legitimate |

## Preprocessing decisions (see `src/anomaly/data.py`)

- **Features used:** `V1`–`V28` + standardized `Amount` (29 features).
- **`Time` is dropped.** It is an arbitrary elapsed-seconds counter, not a
  stationary feature; keeping it would let the model memorize the 2-day
  window rather than learn fraud patterns. Noted as a limitation: a
  production system would engineer proper time-of-day / recency features.
- **Stratified 60/20/20 split** (train/val/test), seed 42.
- **Scaler fit on normal training transactions only** — the scaler defines
  "normal"; fitting on frauds would let anomalies shift the scale.
