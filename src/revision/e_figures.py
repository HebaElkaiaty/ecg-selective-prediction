"""Task 5: Figures 1 (new), 2 and 3, regenerated on the seed-0 fold-8 model.

Vector PDF only -- these go straight into the manuscript, where a rasterised
copy would only lose resolution.

Tier palette validated for CVD separation (worst adjacent-pair OKLab dE = 17.7
under simulated tritanopia; all pairs >= 8 under deuter/protan/tritan).

NOTE: ax.get_legend_handles_labels() silently drops labels beginning
with '_', which would delete _AVB. Handles and labels are collected manually.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.calibrate_acuity import load_tiers

from src.revision.paths import run_dir, FIG as OUT
RUN = run_dir("acuity35")
TIER_C = {"Critical": "#B3261E", "Important": "#E08A1E", "Benign": "#2A6FA8"}
ORD = {"Critical": 0, "Important": 1, "Benign": 2}
INK, MUTED, GRID = "#1a1a1a", "#595959", "#d8d8d8"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})

rows = list(csv.DictReader(open("results/revision/e2_per_class_seed0.csv")))
for r in rows:
    r["n9"] = int(r["n_pos_fold9"]); r["floor"] = int(r["cert_floor"])
    r["alpha"] = float(r["alpha"]); r["fnr"] = float(r["test_fnr"])
    r["flag"] = float(r["flag_rate"]); r["lam"] = float(r["lambda_c"])
    r["lo"] = float(r["fnr_ci95_lo"]); r["hi"] = float(r["fnr_ci95_hi"])
rows.sort(key=lambda r: (ORD[r["tier"]], -r["n9"]))


def fig1():
    """Per-class fold-9 positives against the tier certification floor."""
    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        c = TIER_C[r["tier"]]
        ok = r["n9"] >= r["floor"]
        ax.plot([min(r["n9"], r["floor"]), max(r["n9"], r["floor"])], [yi, yi],
                color=GRID, lw=1.2, zorder=1, solid_capstyle="round")
        ax.scatter(r["floor"], yi, marker="|", s=90, color=MUTED, lw=1.6, zorder=2)
        ax.scatter(r["n9"], yi, s=42, color=c if ok else "white", zorder=3,
                   edgecolor=c, linewidth=1.6)
    ax.set_yticks(y); ax.set_yticklabels([r["class"] for r in rows], fontsize=7.5)
    ax.set_xscale("log"); ax.set_xlim(1.4, 1200)
    ax.set_xlabel("Fold-9 positive examples  (log scale)")
    ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", ls="", mfc=TIER_C[t], mec=TIER_C[t], ms=6, label=t)
               for t in ("Critical", "Important", "Benign")]
    handles += [Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, ms=6,
                       label="below floor (uncertifiable)"),
                Line2D([], [], marker="|", ls="", color=MUTED, ms=9,
                       label="tier certification floor")]
    ax.legend(handles=handles, labels=[h.get_label() for h in handles],
              loc="lower right", frameon=False, fontsize=7.5)
    ax.set_title("Calibration support against the certification floor\n"
                 r"$\delta_k = 0.10/33$; floors 288 / 114 / 56 for $\alpha$ = 0.02 / 0.05 / 0.10",
                 fontsize=9, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_support_vs_floor.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2():
    """Per-class ROC on fold 10, grouped by tier."""
    ts, ty = np.load(RUN/"scores_fold10.npy"), np.load(RUN/"labels_fold10.npy")
    _, order = load_tiers(); col = {c: i for i, c in enumerate(order)}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    for ax, tier in zip(axes, ("Critical", "Important", "Benign")):
        sel = [r for r in rows if r["tier"] == tier]
        pairs = []
        for r in sel:
            j = col[r["class"]]
            if len(np.unique(ty[:, j])) < 2:
                continue
            fpr, tpr, _ = roc_curve(ty[:, j], ts[:, j])
            cert = r["status"] == "certified"
            ln, = ax.plot(fpr, tpr, lw=1.6 if cert else 1.1,
                          ls="-" if cert else (0, (3, 2)),
                          color=TIER_C[tier], alpha=0.95 if cert else 0.45)
            pairs.append((ln, f"{r['class']} ({roc_auc_score(ty[:, j], ts[:, j]):.3f})"))
        ax.plot([0, 1], [0, 1], color=GRID, lw=0.8, zorder=0)
        n_c = sum(1 for r in sel if r["status"] == "certified")
        ax.set_title(f"{tier}  ($\\alpha$ = {sel[0]['alpha']:.2f})\n"
                     f"{n_c} certified · {len(sel)-n_c} uncertifiable",
                     fontsize=8.5, color=TIER_C[tier], loc="left")
        ax.set_xlabel("False positive rate"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.grid(color=GRID, lw=0.5, alpha=0.6); ax.set_axisbelow(True)
        # manual handles: get_legend_handles_labels() would drop _AVB
        ax.legend([p[0] for p in pairs], [p[1] for p in pairs],
                  fontsize=6, loc="lower right", frameon=False, labelspacing=0.25)
    axes[0].set_ylabel("True positive rate")
    fig.suptitle("Per-class ROC, fold 10 (seed 0). Solid: certified at the tier target. "
                 "Dashed: uncertifiable.", fontsize=9, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig2_class_roc.pdf", bbox_inches="tight")
    plt.close(fig)


def fig3():
    """Flag rate (headline) against achieved test FNR for certified classes."""
    cert = [r for r in rows if r["status"] == "certified"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for a, lbl in [(0.02, r"$\alpha$ = 0.02"), (0.05, r"$\alpha$ = 0.05"), (0.10, r"$\alpha$ = 0.10")]:
        ax.axhline(a, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.text(1.005, a, lbl, va="center", fontsize=7, color=MUTED, transform=ax.get_yaxis_transform())
    pairs = []
    for tier in ("Critical", "Important", "Benign"):
        sel = [r for r in cert if r["tier"] == tier]
        if not sel:
            continue
        # 95% Clopper-Pearson interval on the test FNR. Without it a single
        # point above its ceiling reads as a clean failure; IVCD's interval
        # contains alpha, which is the honest reading (R2.2).
        ax.errorbar([r["flag"] for r in sel], [r["fnr"] for r in sel],
                    yerr=[[r["fnr"] - r["lo"] for r in sel],
                          [r["hi"] - r["fnr"] for r in sel]],
                    fmt="none", ecolor=TIER_C[tier], elinewidth=1.1,
                    capsize=2.5, capthick=1.1, alpha=0.75, zorder=2)
        sc = ax.scatter([r["flag"] for r in sel], [r["fnr"] for r in sel],
                        s=58, color=TIER_C[tier], edgecolor="white", linewidth=1.2, zorder=3)
        pairs.append((sc, f"{tier} ($\\alpha$ = {sel[0]['alpha']:.2f})"))
        for r in sel:
            # nudge below when the label would sit on an alpha ceiling line
            near = any(abs(r["fnr"] - a) < 0.006 for a in (0.02, 0.05, 0.10))
            ax.annotate(r["class"], (r["flag"], r["fnr"]), textcoords="offset points",
                        xytext=(6, -9 if near else 4), fontsize=6.5, color=INK)
    ax.set_xlabel("Flag rate  —  fraction of fold-10 records referred for review")
    ax.set_ylabel("Achieved test FNR")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.012, max(r["hi"] for r in cert) * 1.08)
    ax.grid(color=GRID, lw=0.5, alpha=0.6); ax.set_axisbelow(True)
    ax.legend([p[0] for p in pairs], [p[1] for p in pairs], frameon=False,
              fontsize=7.5, loc="upper left")
    ax.set_title("Selectivity against achieved risk, 12 certified classes (seed 0).\n"
                 "A point far to the right meets its target by referring most records; "
                 "bars are 95% Clopper\u2013Pearson intervals on the test FNR.",
                 fontsize=9, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_flagrate_vs_fnr.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    fig1(); fig2(); fig3()
    for p in sorted(OUT.glob("fig[123]_*.pdf")):
        print(f"  {p}  {p.stat().st_size//1024} KB")


if __name__ == "__main__":
    main()
