"""Task 3 (E4): validity simulation with folds 9+10 pooled as the population.

Per trial: draw a calibration set of fold-9 size by PATIENT-level bootstrap from
the pooled population, run the full procedure (33-class family a priori,
delta_k = 0.10/33, exact search over calibration positive scores), then evaluate
each certified class's TRUE FNR on the whole pooled population.

Reported: the family-wise violation rate -- the fraction of TRIALS in which at
least one certified class exceeds its target, which is the quantity the
(1-delta) guarantee bounds -- with a 95% Clopper-Pearson interval, plus
per-class violation rates. The CSV also records the total class-level violation
count and the maximum number of violating classes in any single trial, so the
family-wise figure can never be mistaken for a sum over classes; the two
coincide only when no trial violates in more than one class.
The same bootstrap gives the per-class exceedance rate for CRC, whose guarantee
is on the expectation and therefore is expected to exceed alpha on a fraction of
draws.

Patient-level resampling respects the fact that a patient's records all sit in
one fold, so records are not independent within a patient.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.calibrate_acuity import load_tiers, ALPHA_MAPS
from src.revision.e2_per_class import clopper_pearson, OUT
from src.revision.e_crc import crc_threshold

from src.revision.paths import run_dir
RUN = run_dir("acuity35")
DELTA, N_TRIALS, SEED = 0.10, 2000, 12345


def certify_fast(pos_sorted, alpha, delta_k):
    """Largest observed positive score rejecting H0, else 0. pos_sorted ascending."""
    n = len(pos_sorted)
    if n == 0:
        return 0.0
    # lambda = pos_sorted[k] leaves exactly k misses
    p = stats.binom.cdf(np.arange(n), n, alpha)
    ok = np.nonzero(p <= delta_k)[0]
    return float(pos_sorted[ok[-1]]) if len(ok) else 0.0


def main():
    tiers, order = load_tiers(); col = {c: i for i, c in enumerate(order)}
    A = ALPHA_MAPS["primary"]
    s = np.concatenate([np.load(RUN/"scores_fold9.npy"),  np.load(RUN/"scores_fold10.npy")])
    y = np.concatenate([np.load(RUN/"labels_fold9.npy"),  np.load(RUN/"labels_fold10.npy")])
    pid = np.concatenate([
        np.array([r["patient_id"] for r in csv.DictReader(open(RUN/"meta_fold9.csv"))]),
        np.array([r["patient_id"] for r in csv.DictReader(open(RUN/"meta_fold10.csv"))])])
    n_cal = sum(1 for _ in csv.DictReader(open(RUN/"meta_fold9.csv")))
    print(f"population {s.shape}, {len(np.unique(pid))} patients; calibration size {n_cal}")

    tiered = [(c, t) for c, t in tiers if t != "Excluded"]
    dk = DELTA/len(tiered)
    # population true FNR is a step function of lambda -> precompute sorted positives
    pop_pos = {c: np.sort(s[y[:, col[c]] == 1, col[c]]) for c, _ in tiered}
    true_fnr = lambda c, lam: float(np.searchsorted(pop_pos[c], lam, "left")/len(pop_pos[c])) \
                              if len(pop_pos[c]) else 0.0

    pats, inv = np.unique(pid, return_inverse=True)
    by_pat = [np.nonzero(inv == i)[0] for i in range(len(pats))]
    rng = np.random.default_rng(SEED)

    ltt_v = {c: 0 for c, _ in tiered}; ltt_n = {c: 0 for c, _ in tiered}
    per_trial_violations = []          # how many classes violated in each trial
    crc_v = {c: 0 for c, _ in tiered}; crc_n = {c: 0 for c, _ in tiered}
    fam_viol = 0; fam_trials = 0

    for t in range(N_TRIALS):
        pick = rng.integers(0, len(pats), size=len(pats))
        idx = np.concatenate([by_pat[i] for i in pick])
        if len(idx) > n_cal:
            idx = idx[:n_cal]
        cs, cy = s[idx], y[idx]
        any_v, any_cert, n_v = False, False, 0
        for c, tier in tiered:
            j, a = col[c], A[tier]
            pos = np.sort(cs[cy[:, j] == 1, j])
            lam = certify_fast(pos, a, dk)
            if lam > 0:
                any_cert = True
                ltt_n[c] += 1
                if true_fnr(c, lam) > a:
                    ltt_v[c] += 1; any_v = True; n_v += 1
            lam_c, k = crc_threshold(pos, a)
            if lam_c > 0:
                crc_n[c] += 1
                if true_fnr(c, lam_c) > a:
                    crc_v[c] += 1
        if any_cert:
            fam_trials += 1
            fam_viol += int(any_v)     # TRIALS with >= 1 violation, not classes
            per_trial_violations.append(n_v)
        if (t+1) % 500 == 0:
            print(f"  trial {t+1}/{N_TRIALS}  family-wise violations so far "
                  f"{fam_viol}/{fam_trials}")

    pv = np.array(per_trial_violations)
    lo, hi = clopper_pearson(fam_viol, fam_trials)
    print(f"\n  trials with >=1 violating class : {int((pv >= 1).sum())}  <- family-wise")
    print(f"  total class-level violations    : {int(pv.sum())}")
    print(f"  max violating classes per trial : {int(pv.max()) if len(pv) else 0}")
    print(f"\nFAMILY-WISE violation rate (LTT): {fam_viol}/{fam_trials} = "
          f"{fam_viol/fam_trials:.4f}  95% CI [{lo:.4f}, {hi:.4f}]   target delta = {DELTA}")

    with open(OUT/"e4_validity_simulation.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["scope","class","tier","alpha","n_trials_certified","violations",
                    "violation_rate","ci95_lo","ci95_hi","total_class_level_violations",
                    "max_violations_in_one_trial"])
        # "violations" on this row counts TRIALS with at least one violating class.
        # The two extra columns make that unambiguous: they coincide only when no
        # trial violates in more than one class, which is not guaranteed.
        w.writerow(["family-wise (LTT)","ALL","-","-",fam_trials,fam_viol,
                    round(fam_viol/fam_trials,6),round(lo,6),round(hi,6),
                    int(pv.sum()), int(pv.max()) if len(pv) else 0])
        for scope, V, N in [("per-class (LTT)", ltt_v, ltt_n), ("per-class (CRC)", crc_v, crc_n)]:
            for c, tier in tiered:
                if N[c] == 0: continue
                l2,h2 = clopper_pearson(V[c], N[c])
                w.writerow([scope, c, tier, A[tier], N[c], V[c],
                            round(V[c]/N[c],6), round(l2,6), round(h2,6), "", ""])
    print(f"\n{'class':10s} {'a':>5s} | {'LTT cert':>9s} {'viol':>6s} {'rate':>7s} "
          f"| {'CRC n':>6s} {'viol':>6s} {'rate':>7s}")
    for c, tier in tiered:
        if ltt_n[c] == 0 and crc_n[c] == 0: continue
        lr = ltt_v[c]/ltt_n[c] if ltt_n[c] else float('nan')
        cr = crc_v[c]/crc_n[c] if crc_n[c] else float('nan')
        print(f"{c:10s} {A[tier]:5.2f} | {ltt_n[c]:>9d} {ltt_v[c]:>6d} {lr:7.4f} "
              f"| {crc_n[c]:>6d} {crc_v[c]:>6d} {cr:7.4f}")
    print("\nwrote e4_validity_simulation.csv")


if __name__ == "__main__":
    main()
