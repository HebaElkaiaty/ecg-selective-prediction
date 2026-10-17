"""Task 1: Contribution 2 grid sweep with certified-interval structure.

For each (alpha_c, grid_size): lambda_hat under closest-to-0.5, val/test
inconsistency, corrected mean set size, empty-set rate, and the structure of the
certified set -- how many disjoint intervals, which one contains lambda_hat, and
whether lambda_hat sits at that interval's lower edge (which would make it the
largest non-trivial prediction sets inside that interval).
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.hierarchy import build_hierarchy
from src.calibration.hierarchical_risk_control import build_noisy_or_joint, make_groups
from src.revision.e_contrib2 import incons_and_size, sweep, RUN, DELTA, ALPHAS, GRIDS
from src.revision.paths import OUT


def intervals(mask):
    """Contiguous runs of True as (start_idx, end_idx) inclusive."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j)); i = j + 1
        else:
            i += 1
    return out


def main():
    hier = build_hierarchy(verbose=False)
    groups = make_groups(hier["diagnostic_groups"])
    vj = build_noisy_or_joint(np.load(RUN/"scores_fold9.npy"),  hier["diagnostic_groups"])
    tj = build_noisy_or_joint(np.load(RUN/"scores_fold10.npy"), hier["diagnostic_groups"])

    rows = []
    for gs in GRIDS:
        for a in ALPHAS:
            grid, rates, mask = sweep(vj, groups, a, gs)
            iv = intervals(mask)
            if not iv:
                rows.append(dict(gs=gs, a=a, cert=False)); continue
            cl = grid[mask]
            lam = float(cl[np.argmin(np.abs(cl - 0.5))])
            k = int(np.argmin(np.abs(grid - lam)))
            which = next(w for w, (s, e) in enumerate(iv) if s <= k <= e)
            s, e = iv[which]
            vi = float(rates[k])
            ti, raw, corr, emp = incons_and_size(tj, groups, lam)
            rows.append(dict(gs=gs, a=a, cert=True, lam=lam, vi=vi, ti=ti, raw=raw,
                             corr=corr, emp=emp, n_iv=len(iv), which=which + 1,
                             iv_lo=float(grid[s]), iv_hi=float(grid[e]),
                             at_lower_edge=(k == s), is_upper_iv=(which == len(iv) - 1),
                             n_cert=int(mask.sum())))

    with open(OUT/"e6b_contrib2_grid_detail.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["grid_size","alpha_c","certified","lambda_hat","val_incons_at_lambda",
                    "test_incons","raw_mean_set_size","corrected_mean_set_size",
                    "empty_set_rate","n_certified_intervals","interval_index",
                    "interval_lambda_min","interval_lambda_max","lambda_at_lower_edge",
                    "is_uppermost_interval","n_certified_grid_pts","delta","bonferroni"])
        for r in rows:
            if not r["cert"]:
                w.writerow([r["gs"], r["a"], "no"] + [""]*13 + [DELTA, f"delta/{r['gs']}"]); continue
            w.writerow([r["gs"], r["a"], "yes", round(r["lam"],6), round(r["vi"],6),
                        round(r["ti"],6), round(r["raw"],4), round(r["corr"],4),
                        round(r["emp"],4), r["n_iv"], r["which"], round(r["iv_lo"],6),
                        round(r["iv_hi"],6), "yes" if r["at_lower_edge"] else "no",
                        "yes" if r["is_upper_iv"] else "no", r["n_cert"],
                        DELTA, f"delta/{r['gs']}"])

    print(f"{'grid':>5s} {'a_c':>5s} {'lam':>8s} {'val':>7s} {'test':>7s} {'corr':>6s} "
          f"{'empty':>6s} {'ivs':>4s} {'in#':>4s} {'interval':>19s} {'lower edge':>11s}")
    for r in rows:
        if not r["cert"]:
            print(f"{r['gs']:>5d} {r['a']:>5.2f}   not certified"); continue
        print(f"{r['gs']:>5d} {r['a']:>5.2f} {r['lam']:8.4f} {r['vi']:7.4f} {r['ti']:7.4f} "
              f"{r['corr']:6.2f} {r['emp']:6.3f} {r['n_iv']:>4d} {r['which']:>4d} "
              f"[{r['iv_lo']:.4f},{r['iv_hi']:.4f}] {'YES' if r['at_lower_edge'] else 'no':>11s}")
    print("\nwrote e6b_contrib2_grid_detail.csv")


if __name__ == "__main__":
    main()
