"""Task 2 (E6): hierarchy parent-score baselines, evaluated on fold 10.

Variants
  a) noisy-OR at 0.5, with parent-only flags pruned (deterministic repair)
  b) max-child parent scores at 0.5
  c) super5 (seed-0 fold8) parent scores + 35-class children at 0.5,
     with and without upward closure; plus LTT certification of this pair
  d) logistic aggregation of child scores, fitted on FOLD 8 ONLY

Metrics per variant: inconsistency in both directions, corrected mean set size,
empty-set rate, parent AUROC and ECE, and parent FNR/FPR against the true
superclass labels.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.hierarchy import build_hierarchy, prune_parent_only
from src.calibration.hierarchical_risk_control import build_noisy_or_joint, make_groups
from src.metrics import expected_calibration_error

from src.revision.paths import run_dir, OUT
A35 = run_dir("acuity35")
S5  = run_dir("super5")
DELTA, GRID, ALPHAS = 0.10, 1000, [0.02, 0.05, 0.10]
NORM_SUB, NORM_PAR = 14, 23


def metrics(sub, par, groups, lam, y_super, prune=False, closure=False):
    """sub:(N,23) par:(N,5) -> dict of structural + parent-level metrics."""
    ps, pp = sub >= lam, par >= lam
    if prune:    # deterministic repair of parent-implies-child violations
        pp = prune_parent_only(ps, pp, [{"parent_idx_super": g["p5"],
                                         "child_idx_sub": g["c23"]} for g in groups])
    if closure:  # add the parent whenever any child is flagged
        for g in groups:
            pp[:, g["p5"]] |= ps[:, g["c23"]].any(axis=1)
    c2p = np.zeros(len(sub), bool); p2c = np.zeros(len(sub), bool)
    for g in groups:
        anyc = ps[:, g["c23"]].any(axis=1)
        c2p |= anyc & ~pp[:, g["p5"]]
        p2c |= pp[:, g["p5"]] & ~anyc
    raw = ps.sum(1) + pp.sum(1)
    corr = raw - (ps[:, NORM_SUB] & pp[:, 0]).astype(int)   # NORM pair = one finding
    auroc = float(np.mean([roc_auc_score(y_super[:, k], par[:, k])
                           for k in range(5) if len(np.unique(y_super[:, k])) >= 2]))
    pos = y_super == 1
    fn = (pp & pos).sum() ; tp_pos = pos.sum()
    return dict(c2p=float(c2p.mean()), p2c=float(p2c.mean()),
                any_incons=float((c2p | p2c).mean()),
                raw_size=float(raw.mean()), corr_size=float(corr.mean()),
                empty=float((corr == 0).mean()), parent_auroc=auroc,
                parent_ece=expected_calibration_error(par, y_super, 15),
                parent_fnr=float(((~pp) & pos).sum() / pos.sum()),
                parent_fpr=float((pp & ~pos).sum() / (~pos).sum()))


def ltt_certify(sub, par, groups, alpha):
    """LTT over a 1000-point grid with grid-wide Bonferroni (R is non-monotone)."""
    grid = np.linspace(0, 1, GRID); N = len(sub)
    inc = np.zeros((N, GRID), bool)
    ps = sub[:, :, None] >= grid[None, None, :]
    pp = par[:, :, None] >= grid[None, None, :]
    for g in groups:
        anyc = ps[:, g["c23"], :].any(axis=1); p = pp[:, g["p5"], :]
        inc |= (anyc & ~p) | (p & ~anyc)
    k = inc.sum(0).astype(int)
    mask = stats.binom.cdf(k, N, alpha) <= DELTA / GRID
    if not mask.any():
        return None, k / N
    cl = grid[mask]
    return float(cl[np.argmin(np.abs(cl - 0.5))]), k / N


def main():
    hier = build_hierarchy(verbose=False)
    groups = [{"p5": g["parent_idx_super"], "c23": g["child_idx_sub"],
               "name": g["parent"]} for g in hier["diagnostic_groups"]]
    a8  = np.load(A35/"scores_fold8.npy");  a9 = np.load(A35/"scores_fold9.npy")
    a10 = np.load(A35/"scores_fold10.npy")
    # Map the 5-class model's columns into SUPERDIAG_ORDER BY NAME. The model
    # emits alphabetical order (CD, HYP, MI, NORM, STTC); SUPERDIAG_ORDER is
    # NORM, MI, STTC, CD, HYP. Positional use silently mis-pairs every group.
    from src.utils.hierarchy import SUPERDIAG_ORDER, super_column_index
    s5_order = (S5/"class_order.txt").read_text().split()
    remap = super_column_index(s5_order)
    print(f"  super5 columns {s5_order} -> SUPERDIAG_ORDER {SUPERDIAG_ORDER} via {remap}")
    s8  = np.load(S5/"scores_fold8.npy")[:, remap]
    s10 = np.load(S5/"scores_fold10.npy")[:, remap]
    y10 = np.load(S5/"labels_fold10.npy")[:, remap]
    sub8, sub10 = a8[:, :23], a10[:, :23]

    nor10 = build_noisy_or_joint(a10, hier["diagnostic_groups"])[:, 23:]
    max10 = np.column_stack([sub10[:, g["c23"]].max(axis=1) for g in groups])

    # (d) logistic aggregation fitted on fold 8 only
    idx8 = np.load("checkpoints/retrain/fold8_index.npy")
    import sys as _s; _s.path.insert(0, ".")
    from src.utils.data import load_ptbxl_metadata, aggregate_labels
    from config import PTBXL_PATH
    import pandas as pd
    meta = load_ptbxl_metadata(PTBXL_PATH)
    scp = pd.read_csv(Path(PTBXL_PATH)/"scp_statements.csv", index_col=0)
    lab = aggregate_labels(meta, scp, task="super")
    cols = [c for c in lab.columns if c not in list(meta.columns)+["label_primary"]]
    assert cols == s5_order, f"label col order drift: {cols} vs {s5_order}"
    Y5 = lab[cols].values.astype(np.float32)[:, remap]; y8 = Y5[idx8]
    log10 = np.zeros_like(max10)
    for g in groups:
        X8 = sub8[:, g["c23"]]
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X8, y8[:, g["p5"]])
        log10[:, g["p5"]] = clf.predict_proba(sub10[:, g["c23"]])[:, 1]

    rows = []
    def add(name, par, **kw):
        rows.append((name, metrics(sub10, par, groups, 0.5, y10, **kw)))

    add("noisy-OR @0.5", nor10)
    add("noisy-OR @0.5 + parent-only pruning", nor10, prune=True)
    add("max-child @0.5", max10)
    add("super5 parent @0.5", s10)
    add("super5 parent @0.5 + upward closure", s10, closure=True)
    add("logistic (fit fold 8) @0.5", log10)

    with open(OUT/"e6c_hierarchy_baselines.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["variant","child_implies_parent_rate","parent_implies_child_rate",
                    "any_inconsistency","raw_mean_set_size","corrected_mean_set_size",
                    "empty_set_rate","parent_auroc","parent_ece","parent_fnr","parent_fpr"])
        for n,m in rows:
            w.writerow([n]+[round(m[k],6) for k in
                ("c2p","p2c","any_incons","raw_size","corr_size","empty",
                 "parent_auroc","parent_ece","parent_fnr","parent_fpr")])
    print(f"{'variant':38s} {'c->p':>7s} {'p->c':>7s} {'any':>7s} {'corr':>6s} "
          f"{'empty':>6s} {'pAUROC':>7s} {'pECE':>7s} {'pFNR':>6s} {'pFPR':>6s}")
    for n,m in rows:
        print(f"{n:38s} {m['c2p']:7.4f} {m['p2c']:7.4f} {m['any_incons']:7.4f} "
              f"{m['corr_size']:6.2f} {m['empty']:6.3f} {m['parent_auroc']:7.4f} "
              f"{m['parent_ece']:7.4f} {m['parent_fnr']:6.3f} {m['parent_fpr']:6.3f}")

    # (c) LTT certification of the two-model pair
    s9 = np.load(S5/"scores_fold9.npy")[:, remap]; sub9 = a9[:, :23]
    print("\nLTT certification, super5 parents + 35-class children (two independent models):")
    with open(OUT/"e6c_super5_pair_ltt.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["alpha_c","certified","lambda_hat","val_incons_at_lambda","test_incons",
                    "corrected_mean_set_size","empty_set_rate","delta","bonferroni"])
        for a in ALPHAS:
            lam, rates = ltt_certify(sub9, s9, groups, a)
            if lam is None:
                w.writerow([a,"no","","","","","",DELTA,f"delta/{GRID}"])
                print(f"  alpha_c={a:.2f}: not certified"); continue
            m = metrics(sub10, s10, groups, lam, y10)
            vi = float(rates[int(round(lam*(GRID-1)))])
            w.writerow([a,"yes",round(lam,6),round(vi,6),round(m['any_incons'],6),
                        round(m['corr_size'],4),round(m['empty'],4),DELTA,f"delta/{GRID}"])
            print(f"  alpha_c={a:.2f}: lambda={lam:.4f} val={vi:.4f} test={m['any_incons']:.4f} "
                  f"corr_size={m['corr_size']:.2f} empty={m['empty']:.3f} "
                  f"(c->p {m['c2p']:.4f}, p->c {m['p2c']:.4f})")
    print("\nwrote e6c_hierarchy_baselines.csv, e6c_super5_pair_ltt.csv")


if __name__ == "__main__":
    main()
