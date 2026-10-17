"""R4.6: estimator power/validity trade-off.  R3 Q1: the selection-dependence arm.

Part A -- estimators. Hoeffding and exact-binomial p-values, each under the
monotone-boundary search and under grid-wide Bonferroni, compared on (i) how
many classes they certify on the real calibration fold and (ii) the family-wise
violation rate under the patient bootstrap. This is the power/validity trade-off
R4.6 asks to see stated rather than implied.

Part B -- selection dependence. The submitted design chose the checkpoint on the
same fold used for calibration. The revised design does not, but R3 Q1 asks
whether the simulation accounts for that dependence. Here it does, directly:
in the DEPENDENT arm each trial picks, among the five seeds, the one with the
best macro AUROC on that trial's own calibration draw, then calibrates and is
evaluated using that seed. The INDEPENDENT arm fixes seed 0 a priori. The gap
between the two is the optimism the submitted design carried.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.calibrate_acuity import load_tiers, ALPHA_MAPS
from src.revision.paths import run_dir, OUT, SEEDS
from src.revision.e2_per_class import clopper_pearson
from src.revision.e4_validity_sim import SEED as _VALIDITY_SEED

# SEED matches e4_validity_sim deliberately. Both draw the same patient-level
# bootstrap in the same order, so the independent arm here reproduces the
# validity study's family-wise violation rate exactly rather than giving a
# second, slightly different Monte Carlo estimate of the same quantity. The
# paper quotes one number; e4_validity_sim is its source.
DELTA, N_TRIALS, SEED, GRID = 0.10, 2000, 12345, 1000


def fast_macro_auc(scores, labels):
    """Macro AUROC via the rank identity, vectorised over all classes at once.

    AUC = (sum of positive ranks - n_p(n_p+1)/2) / (n_p * n_n). Equivalent to
    sklearn's roc_auc_score (ties averaged) but ~50x faster here, which matters
    at 5 seeds x 2,000 trials.
    """
    r = rankdata(scores, axis=0)
    npos = labels.sum(axis=0); nneg = labels.shape[0] - npos
    ok = (npos > 0) & (nneg > 0)
    rsum = (r * labels).sum(axis=0)
    auc = (rsum - npos * (npos + 1) / 2.0) / np.maximum(npos * nneg, 1)
    return float(auc[ok].mean()) if ok.any() else float("nan")


assert SEED == _VALIDITY_SEED, (
    f"bootstrap seed {SEED} differs from the validity study's {_VALIDITY_SEED}; the "
    "independent arm would then be a second, slightly different Monte Carlo estimate "
    "of the same quantity and the paper would carry two family-wise violation rates"
)


def p_values(pos_sorted, cand, n, alpha, method):
    k = np.searchsorted(pos_sorted, cand, side="left")
    if method == "binomial":
        return stats.binom.cdf(k, n, alpha)
    return np.exp(-2.0 * n * np.maximum(0.0, alpha - k / n) ** 2)


def lam_for(pos, alpha, delta_k, method, grid_bonf):
    n = len(pos)
    if n == 0:
        return 0.0
    cand = np.linspace(0, 1, GRID) if grid_bonf else np.unique(np.concatenate([[0.0], pos]))
    lvl = delta_k / GRID if grid_bonf else delta_k
    p = p_values(np.sort(pos), cand, n, alpha, method)
    ok = np.nonzero(p <= lvl)[0]
    return float(cand[ok[-1]]) if len(ok) else 0.0


def main():
    tiers, order = load_tiers(); col = {c: i for i, c in enumerate(order)}
    A = ALPHA_MAPS["primary"]
    tiered = [(c, t) for c, t in tiers if t != "Excluded"]
    m = len(tiered); dk = DELTA / m

    per_seed = {}
    for s in SEEDS:
        d = run_dir("acuity35", seed=s)
        per_seed[s] = (np.concatenate([np.load(d/"scores_fold9.npy"), np.load(d/"scores_fold10.npy")]),
                       np.concatenate([np.load(d/"labels_fold9.npy"), np.load(d/"labels_fold10.npy")]))
    s0, y0 = per_seed[SEEDS[0]]
    pid = np.concatenate([
        np.array([r["patient_id"] for r in csv.DictReader(open(run_dir("acuity35")/"meta_fold9.csv"))]),
        np.array([r["patient_id"] for r in csv.DictReader(open(run_dir("acuity35")/"meta_fold10.csv"))])])
    n_cal = sum(1 for _ in csv.DictReader(open(run_dir("acuity35")/"meta_fold9.csv")))

    VARIANTS = [("binomial", False, "Binomial + monotone boundary (adopted)"),
                ("binomial", True,  "Binomial + grid-Bonferroni"),
                ("hoeffding", False, "Hoeffding + monotone boundary"),
                ("hoeffding", True,  "Hoeffding + grid-Bonferroni")]

    # ---- Part A(i): certifications on the real calibration fold ------------
    real = {}
    for meth, gb, label in VARIANTS:
        cert = []
        for c, tier in tiered:
            j, a = col[c], A[tier]
            pos = s0[:n_cal, j][y0[:n_cal, j] == 1]
            if lam_for(pos, a, dk, meth, gb) > 0:
                cert.append(c)
        real[label] = cert
        print(f"  {label:42s} certifies {len(cert):2d}/33 on fold 9")

    pats, inv = np.unique(pid, return_inverse=True)
    by_pat = [np.nonzero(inv == i)[0] for i in range(len(pats))]
    pop_pos = {c: {s: np.sort(per_seed[s][0][per_seed[s][1][:, col[c]] == 1, col[c]])
                   for s in SEEDS} for c, _ in tiered}

    def true_fnr(c, s, lam):
        p = pop_pos[c][s]
        return float(np.searchsorted(p, lam, "left") / len(p)) if len(p) else 0.0

    # ---- Part A(ii) + Part B: bootstrap ------------------------------------
    # An estimator that certifies nothing on the real fold has no operating
    # point to violate; bootstrapping it would only divide by zero.
    ACTIVE = [(m_, g_, l_) for m_, g_, l_ in VARIANTS if real[l_]]
    print(f"\n  bootstrapping {len(ACTIVE)} of {len(VARIANTS)} estimators "
          f"(the others certify nothing)")
    rng = np.random.default_rng(SEED)
    arms = {label: 0 for _, _, label in VARIANTS}
    arms_n = {label: 0 for _, _, label in VARIANTS}
    dep_v = dep_n = ind_v = ind_n = 0
    sel_counts = {s: 0 for s in SEEDS}

    for _ in range(N_TRIALS):
        pick = rng.integers(0, len(pats), size=len(pats))
        idx = np.concatenate([by_pat[i] for i in pick])[:n_cal]

        for meth, gb, label in ACTIVE:
            any_v = any_c = False
            for c, tier in tiered:
                j, a = col[c], A[tier]
                pos = s0[idx, j][y0[idx, j] == 1]
                lam = lam_for(pos, a, dk, meth, gb)
                if lam > 0:
                    any_c = True
                    if true_fnr(c, SEEDS[0], lam) > a:
                        any_v = True
            if any_c:
                arms_n[label] += 1; arms[label] += int(any_v)

        # Part B -- dependent arm: pick the seed that looks best on THIS draw
        best, best_auc = SEEDS[0], -1.0
        for s in SEEDS:
            sc, yy = per_seed[s]
            au = fast_macro_auc(sc[idx], yy[idx])
            if au > best_auc:
                best_auc, best = au, s
        sel_counts[best] += 1
        for s, bucket in ((best, "dep"), (SEEDS[0], "ind")):
            sc, yy = per_seed[s]
            any_v = any_c = False
            for c, tier in tiered:
                j, a = col[c], A[tier]
                pos = sc[idx, j][yy[idx, j] == 1]
                lam = lam_for(pos, a, dk, "binomial", False)
                if lam > 0:
                    any_c = True
                    if true_fnr(c, s, lam) > a:
                        any_v = True
            if any_c:
                if bucket == "dep": dep_n += 1; dep_v += int(any_v)
                else:               ind_n += 1; ind_v += int(any_v)

    with open(OUT/"e10_estimator_variants.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["estimator","n_certified_fold9","certified_classes","trials",
                    "family_wise_violations","violation_rate","ci95_lo","ci95_hi","delta"])
        for _, _, label in VARIANTS:
            n = arms_n[label]
            if n == 0:
                w.writerow([label, len(real[label]), " ".join(sorted(real[label])),
                            0, "", "", "", "", DELTA]); continue
            lo, hi = clopper_pearson(arms[label], n)
            w.writerow([label, len(real[label]), " ".join(sorted(real[label])), n,
                        arms[label], round(arms[label]/n, 6),
                        round(lo, 6), round(hi, 6), DELTA])
    print(f"\n{'estimator':42s} {'cert':>5s} {'FW violation':>13s} {'95% CI':>18s}")
    for _, _, label in VARIANTS:
        n = arms_n[label]
        if n == 0:
            print(f"  {label:42s} {len(real[label]):>3d}  certifies nothing -- no operating point")
            continue
        lo, hi = clopper_pearson(arms[label], n)
        print(f"  {label:42s} {len(real[label]):>3d}  {arms[label]:>4d}/{n:<5d} "
              f"{arms[label]/n:.4f} [{lo:.4f},{hi:.4f}]")

    with open(OUT/"e10_selection_dependence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm","description","trials","family_wise_violations","violation_rate",
                    "ci95_lo","ci95_hi","delta"])
        for arm, desc, v, n in [
            ("independent", "seed fixed a priori (the revised design)", ind_v, ind_n),
            ("dependent", "seed chosen by AUROC on the same draw used to calibrate", dep_v, dep_n)]:
            lo, hi = clopper_pearson(v, n)
            w.writerow([arm, desc, n, v, round(v/n, 6), round(lo, 6), round(hi, 6), DELTA])
    li, hi_ = clopper_pearson(ind_v, ind_n); ld, hd = clopper_pearson(dep_v, dep_n)
    print(f"\nSelection dependence (family-wise violation rate, delta = {DELTA}):")
    print(f"  independent (seed fixed a priori)        {ind_v:>4d}/{ind_n:<5d} "
          f"{ind_v/ind_n:.4f} [{li:.4f},{hi_:.4f}]")
    print(f"  dependent  (seed picked on the same draw) {dep_v:>4d}/{dep_n:<5d} "
          f"{dep_v/dep_n:.4f} [{ld:.4f},{hd:.4f}]")
    print(f"  seed chosen in the dependent arm: {sel_counts}")
    print("\nwrote e10_estimator_variants.csv, e10_selection_dependence.csv")


if __name__ == "__main__":
    main()
