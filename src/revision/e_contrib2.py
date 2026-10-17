"""Task 6: Contribution 2 (noisy-OR hierarchy consistency) on the seed-0 fold-8 model.

Refreshes Table 8 and Figure 4. Adds:
  - grid-size check at 100 / 250 / 1000 candidate thresholds (R3 Q3: this grid
    DOES carry a grid-wide Bonferroni correction, delta/grid_size, because
    R(lambda) is non-monotone -- unlike Contribution 1's exact monotone search)
  - two selection rules: certified lambda closest to 0.5, and smallest certified lambda
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.hierarchy import build_hierarchy
from src.calibration.hierarchical_risk_control import build_noisy_or_joint, make_groups

from src.revision.paths import run_dir, OUT, FIG
RUN = run_dir("acuity35")
DELTA, ALPHAS, GRIDS = 0.10, [0.02, 0.05, 0.10], [100, 250, 1000]
NORM_SUB, NORM_PAR = 14, 23


def incons_and_size(joint, groups, lam):
    pred = joint >= lam
    inc = np.zeros(len(joint), bool)
    for g in groups:
        p = pred[:, g["parent_col"]]; ch = pred[:, g["child_cols"]].any(axis=1)
        inc |= (ch & ~p) | (p & ~ch)
    raw = pred.sum(axis=1)
    # NORM_sub and NORM_par always co-fire (noisy-OR over a single child);
    # the pair is one clinical finding.
    corrected = raw - (pred[:, NORM_SUB] & pred[:, NORM_PAR]).astype(int)
    return float(inc.mean()), float(raw.mean()), float(corrected.mean()), float((corrected == 0).mean())


def sweep(joint, groups, alpha, grid_size):
    grid = np.linspace(0.0, 1.0, grid_size)
    N = len(joint)
    pred = joint[:, :, None] >= grid[None, None, :]
    inc = np.zeros((N, grid_size), bool)
    for g in groups:
        p = pred[:, g["parent_col"], :]; ch = pred[:, g["child_cols"], :].any(axis=1)
        inc |= (ch & ~p) | (p & ~ch)
    k = inc.sum(axis=0).astype(int)
    pv = stats.binom.cdf(k, N, alpha)
    mask = pv <= DELTA / grid_size          # grid-wide Bonferroni (non-monotone R)
    return grid, k / N, mask


def main():
    hier = build_hierarchy(verbose=False)
    groups = make_groups(hier["diagnostic_groups"])
    vj = build_noisy_or_joint(np.load(RUN/"scores_fold9.npy"),  hier["diagnostic_groups"])
    tj = build_noisy_or_joint(np.load(RUN/"scores_fold10.npy"), hier["diagnostic_groups"])
    print(f"joint matrices: val {vj.shape} test {tj.shape}")

    b_v = incons_and_size(vj, groups, 0.5); b_t = incons_and_size(tj, groups, 0.5)
    print(f"baseline lambda=0.5: val incons {b_v[0]:.4f}  test incons {b_t[0]:.4f}  "
          f"corrected set size {b_t[2]:.2f}")

    rows = []
    for gs in GRIDS:
        for a in ALPHAS:
            grid, rates, mask = sweep(vj, groups, a, gs)
            if not mask.any():
                rows.append(dict(grid=gs, alpha=a, rule="-", cert=False)); continue
            cl = grid[mask]
            for rule, lam in [("closest_to_0.5", float(cl[np.argmin(np.abs(cl-0.5))])),
                              ("smallest_certified", float(cl.min()))]:
                ti, traw, tcorr, temp = incons_and_size(tj, groups, lam)
                vi = float(rates[np.argmin(np.abs(grid-lam))])
                rows.append(dict(grid=gs, alpha=a, rule=rule, cert=True, lam=lam,
                                 val_incons=vi, test_incons=ti, raw=traw,
                                 corrected=tcorr, empty=temp,
                                 n_cert=int(mask.sum()),
                                 lo=float(cl.min()), hi=float(cl.max())))

    with open(OUT/"e6_contrib2_table8.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["grid_size","alpha_c","selection_rule","certified","lambda_c",
                    "val_incons_at_lambda","test_incons","raw_mean_set_size",
                    "corrected_mean_set_size","empty_set_rate","n_certified_grid_pts",
                    "certified_lambda_min","certified_lambda_max","delta","bonferroni"])
        w.writerow([ "-","-","baseline lambda=0.5","-",0.5,round(b_v[0],6),round(b_t[0],6),
                     round(b_t[1],4),round(b_t[2],4),round(b_t[3],4),"-","-","-",DELTA,"-"])
        for r in rows:
            if not r["cert"]:
                w.writerow([r["grid"],r["alpha"],"-","no"]+[""]*9+[DELTA,f"delta/{r['grid']}"]); continue
            w.writerow([r["grid"],r["alpha"],r["rule"],"yes",round(r["lam"],6),
                        round(r["val_incons"],6),round(r["test_incons"],6),
                        round(r["raw"],4),round(r["corrected"],4),round(r["empty"],4),
                        r["n_cert"],round(r["lo"],6),round(r["hi"],6),DELTA,f"delta/{r['grid']}"])
    for r in rows:
        if r["cert"] and r["grid"]==1000:
            print(f"  grid={r['grid']:4d} a={r['alpha']:.2f} {r['rule']:19s} "
                  f"lam={r['lam']:.4f} test_incons={r['test_incons']:.4f} "
                  f"corrected_size={r['corrected']:.2f} empty={r['empty']:.3f}")

    # Figure 4
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    grid, rates, _ = sweep(vj, groups, 0.05, 1000)
    ax.plot(grid, rates*100, color="#1a1a1a", lw=1.4, label=r"$R(\lambda)$, fold 9")
    ax.axvline(0.5, color="#595959", lw=1.0, label=r"$\lambda = 0.5$ (baseline)")
    cols = {0.02:"#B3261E", 0.05:"#E08A1E", 0.10:"#2A6FA8"}
    pairs=[]
    for a in ALPHAS:
        _,_,m = sweep(vj, groups, a, 1000)
        if m.any():
            cl=grid[m]; lam=float(cl[np.argmin(np.abs(cl-0.5))])
            ln=ax.axvline(lam, color=cols[a], lw=1.3, ls=(0,(4,2)))
            pairs.append((ln, rf"$\hat\lambda$={lam:.3f} ($\alpha_c$={a})"))
        ax.axhline(a*100, color=cols[a], lw=0.7, ls=(0,(1,3)))
    h0,=ax.plot([],[],color="#1a1a1a",lw=1.4); h1=ax.axvline(0.5,color="#595959",lw=1.0)
    # outside the axes: the in-plot legend covered the certified thresholds
    ax.legend([h0,h1]+[p[0] for p in pairs],
              [r"$R(\lambda)$, fold 9", r"$\lambda=0.5$ (baseline)"]+[p[1] for p in pairs],
              frameon=False, fontsize=7.5, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.set_xlabel(r"Threshold $\lambda$"); ax.set_ylabel("Inconsistency rate $R(\\lambda)$  [%]")
    ax.set_xlim(0,1); ax.set_ylim(0, max(rates*100)*1.15)
    ax.grid(color="#d8d8d8", lw=0.5, alpha=0.6); ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_title("Fold-9 hierarchical inconsistency under noisy-OR (seed 0).\n"
                 "Non-monotone in $\\lambda$, so grid-wide Bonferroni ($\\delta/1000$) is required.",
                 fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(FIG/"fig4_inconsistency_curve.pdf", bbox_inches="tight")
    print("wrote e6_contrib2_table8.csv, fig4_inconsistency_curve.pdf")


if __name__ == "__main__":
    main()
