"""Download the ULB credit-card fraud dataset from the public mirror and verify it.

Usage:
    python data/download.py
"""

from __future__ import annotations

import csv
import sys
import urllib.request
from pathlib import Path

MIRROR_URL = (
    "https://raw.githubusercontent.com/nsethi31/"
    "Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
)
ORIGINAL = "Kaggle dataset mlg-ulb/creditcardfraud (ULB Machine Learning Group)"
EXPECTED_ROWS = 284_807
EXPECTED_COLS = 31
EXPECTED_FRAUDS = 492


def main() -> None:
    out = Path(__file__).resolve().parent / "creditcard.csv"
    if out.exists():
        print(f"{out} already present — verifying ...")
    else:
        print(f"Downloading from {MIRROR_URL} ...")
        req = urllib.request.Request(MIRROR_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r, open(out, "wb") as f:
            f.write(r.read())
        print("Download complete.")

    with open(out, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_rows = sum(1 for _ in reader)
    assert (n_rows, len(header)) == (EXPECTED_ROWS, EXPECTED_COLS), (
        f"shape mismatch: got {(n_rows, len(header))}"
    )
    assert header[0] == "Time" and header[-1] == "Class", "unexpected header"

    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        n_fraud = sum(1 for row in reader if row["Class"] == "1")
    assert n_fraud == EXPECTED_FRAUDS, f"fraud count mismatch: {n_fraud}"
    print(f"Verified: {n_rows} rows x {len(header)} cols, {n_fraud} frauds. OK.")


if __name__ == "__main__":
    sys.exit(main())
