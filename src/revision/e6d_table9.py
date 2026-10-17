"""Table 9: hierarchy-consistency methods compared on fold 10 (seed 0).

Contribution 2 is reported as a COMPARISON of hierarchy-consistency methods,
not as a proposed method. Exact inconsistent-record counts out of 2,198 with
95% Clopper-Pearson intervals.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.hierarchy import build_hierarchy, prune_parent_only, super_column_index
from src.calibration.hierarchical_risk_control import (
    build_noisy_or_joint, make_groups, calibrate_inconsistency_threshold)
from src.revision.e2_per_class import clopper_pearson, OUT

from src.revision.paths import run_dir
A35 = run_dir("acuity35")
S5  = run_dir("super5")
DELTA, GRID, ALPHAS = 0.10, 1000, [0.10, 0.05, 0.02]
NORM_SUB, NORM_PAR = 14, 23


def counts(sub, par, groups, lam, prune=False):
    ps, pp = sub >= lam, par >= lam
    if prune:
        pp = prune_parent_only(ps, pp, groups)
    c2p = np.zeros(len(sub), bool); p2c = np.zeros(len(sub), bool)
    for g in groups:
        anyc = ps[:, g["child_idx_sub"]].any(axis=1); p = pp[:, g["parent_idx_super"]]
        c2p |= anyc & ~p; p2c |= p & ~anyc
    inc = c2p | p2c
    size = ps.sum(1) + pp.sum(1) - (ps[:, NORM_SUB] & pp[:, 0])
    n = len(sub); k = int(inc.sum()); lo, hi = clopper_pearson(k, n)
    return dict(n=n, k=k, rate=k/n, lo=lo, hi=hi, c2p=int(c2p.sum()), p2c=int(p2c.sum()),
                size=float(size.mean()), empty=float((size == 0).mean()))


def main():
    hier = build_hierarchy(verbose=False)
    groups = hier["diagnostic_groups"]
    gj = make_groups(groups)
    a9, a10 = np.load(A35/"scores_fold9.npy"), np.load(A35/"scores_fold10.npy")
    vj = build_noisy_or_joint(a9, groups)
    sub9, sub10 = a9[:, :23], a10[:, :23]
    nor9, nor10 = vj[:, 23:], build_noisy_or_joint(a10, groups)[:, 23:]
    rm = super_column_index(S5/"class_order.txt")
    s9, s10 = np.load(S5/"scores_fold9.npy")[:, rm], np.load(S5/"scores_fold10.npy")[:, rm]

    rows = []
    def add(label, sub, par, lam, prune=False, cert="n/a", nouse="n/a", alpha=""):
        c = counts(sub, par, groups, lam, prune)
        rows.append(dict(label=label, alpha=alpha, lam=lam, cert=cert, nouse=nouse, **c))

    add("noisy-OR, lambda = 0.5 (uncalibrated)", sub10, nor10, 0.5)
    add("noisy-OR + deterministic pruning, lambda = 0.5 (recommended)",
        sub10, nor10, 0.5, prune=True)
    for a in ALPHAS:
        r = calibrate_inconsistency_threshold(vj, gj, alpha=a, delta=DELTA, grid_size=GRID)
        add(f"noisy-OR, certified global threshold", sub10, nor10, r["lambda_c"],
            cert="yes" if r["certified"] else "no",
            nouse="yes" if r["no_useful_certificate"] else "no", alpha=a)

    # five-class pair: certify on fold 9, evaluate on fold 10
    grid = np.linspace(0, 1, GRID)
    from scipy import stats
    inc = np.zeros((len(sub9), GRID), bool)
    ps = sub9[:, :, None] >= grid[None, None, :]
    pp = s9[:, :, None] >= grid[None, None, :]
    for g in groups:
        anyc = ps[:, g["child_idx_sub"], :].any(axis=1); p = pp[:, g["parent_idx_super"], :]
        inc |= (anyc & ~p) | (p & ~anyc)
    k = inc.sum(0).astype(int)
    mask = stats.binom.cdf(k, len(sub9), 0.10) <= DELTA/GRID
    if mask.any():
        cl = grid[mask]; lam = float(cl[np.argmin(np.abs(cl-0.5))])
        emp = counts(sub9, s9, groups, lam)["empty"]
        add("five-class model as parent, certified global threshold", sub10, s10, lam,
            cert="yes", nouse="yes" if (lam <= 0 or emp >= 0.99) else "no", alpha=0.10)
    add("five-class model as parent, lambda = 0.5 (uncalibrated)", sub10, s10, 0.5)

    with open(OUT/"e6_table9_consistency_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method","alpha_c","lambda","certified","no_useful_certificate",
                    "n_test_records","n_inconsistent","inconsistency_rate",
                    "ci95_lo","ci95_hi","n_child_implies_parent","n_parent_implies_child",
                    "corrected_mean_set_size","empty_set_rate"])
        for r in rows:
            w.writerow([r["label"], r["alpha"], round(r["lam"],6), r["cert"], r["nouse"],
                        r["n"], r["k"], round(r["rate"],6), round(r["lo"],6), round(r["hi"],6),
                        r["c2p"], r["p2c"], round(r["size"],4), round(r["empty"],4)])

    print(f"{'method':58s} {'a_c':>5s} {'lam':>7s} {'incons/2198':>12s} {'rate':>7s} "
          f"{'95% CI':>16s} {'size':>5s} {'empty':>6s} {'no-use':>7s}")
    for r in rows:
        print(f"{r['label']:58s} {str(r['alpha']):>5s} {r['lam']:7.4f} "
              f"{r['k']:>6d}/{r['n']:<5d} {r['rate']:7.4f} "
              f"[{r['lo']:.4f},{r['hi']:.4f}] {r['size']:5.2f} {r['empty']:6.3f} {r['nouse']:>7s}")
    print("\nwrote e6_table9_consistency_comparison.csv")


if __name__ == "__main__":
    main()
