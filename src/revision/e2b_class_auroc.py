"""Per-class AUROC on fold 10 for the seed-0 fold-8 35-class model.

Discrimination for all 33 tiered classes, alongside certification status, so the
two can be read against each other: certifiability is governed by calibration
support, not by how well a class separates.

Uses the same computation as the Figure 2 legend (sklearn roc_auc_score on the
fold-10 scores), and asserts agreement with it.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.calibrate_acuity import load_tiers
from src.revision.paths import run_dir, OUT

ORD = {"Critical": 0, "Important": 1, "Benign": 2}


def per_class_auroc():
    tiers, order = load_tiers()
    col = {c: i for i, c in enumerate(order)}
    d = run_dir("acuity35")
    ts, ty = np.load(d / "scores_fold10.npy"), np.load(d / "labels_fold10.npy")
    out = {}
    for c, tier in tiers:
        if tier == "Excluded":
            continue
        j = col[c]
        y = ty[:, j]
        out[c] = (tier, float(roc_auc_score(y, ts[:, j])) if len(np.unique(y)) >= 2
                  else float("nan"), int(y.sum()))
    return out


def main():
    auroc = per_class_auroc()
    status = {r["class"]: r["status"]
              for r in csv.DictReader(open(OUT / "e2_per_class_seed0.csv"))}

    rows = sorted(auroc.items(), key=lambda kv: (ORD[kv[1][0]], kv[0]))
    with open(OUT / "e2_class_auroc_seed0.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "tier", "certified", "auroc"])
        for c, (tier, a, _) in rows:
            w.writerow([c, tier, "yes" if status.get(c) == "certified" else "no", round(a, 6)])

    print(f"{'class':10s} {'tier':10s} {'cert':>5s} {'n+ f10':>7s} {'AUROC':>7s}")
    for c, (tier, a, n) in rows:
        print(f"{c:10s} {tier:10s} {'yes' if status.get(c)=='certified' else 'no':>5s} "
              f"{n:>7d} {a:7.4f}")

    cert = [(c, a) for c, (t, a, _) in auroc.items() if status.get(c) == "certified"]
    lo = min(cert, key=lambda x: x[1])
    print(f"\n  certified classes: {len(cert)}   "
          f"AUROC range {min(a for _, a in cert):.4f}-{max(a for _, a in cert):.4f}")
    print(f"  lowest-AUROC certified class: {lo[0]} ({lo[1]:.4f})")
    print(f"\nwrote e2_class_auroc_seed0.csv")


if __name__ == "__main__":
    main()
