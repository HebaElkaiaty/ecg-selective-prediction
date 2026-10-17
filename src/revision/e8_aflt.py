"""E8: AFLT positive counts by fold (R1 asks for folds 1-7 specifically)."""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PTBXL_PATH
from src.utils.data import load_ptbxl_metadata, get_fold_splits
from src.retrain import build_labels
from src.revision.paths import OUT

CLASS = "AFLT"


def main():
    meta = load_ptbxl_metadata(PTBXL_PATH)
    splits = get_fold_splits(meta)
    Y, names = build_labels("acuity35", meta, splits)
    j = names.index(CLASS)
    folds = meta["strat_fold"].values
    rows = []
    for label, m in [("folds 1-7", folds <= 7), ("fold 8", folds == 8),
                     ("folds 1-8", folds <= 8), ("fold 9 (cal)", folds == 9),
                     ("fold 10 (test)", folds == 10), ("all", folds > 0)]:
        rows.append((label, int(Y[m, j].sum()), int(m.sum())))
        print(f"  {CLASS} positives, {label:16s}: {rows[-1][1]:4d}  (of {rows[-1][2]:5d} records)")
    with open(OUT/"e8_aflt_support.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["split", "aflt_positives", "n_records"]); w.writerows(rows)
    print("wrote e8_aflt_support.csv")


if __name__ == "__main__":
    main()
