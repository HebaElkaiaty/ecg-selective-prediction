"""Table S5 (R2.3): class-conditional CRC at tiered and three uniform targets.

Aggregates the per-class CRC results already produced by e_crc.py -- same
implementation, seed 0, fold 9 calibration, fold 10 test -- into one comparison
across four configurations, and adds a resampling exceedance rate from the same
patient-level bootstrap the validity study uses (seed 12345).

Definitions, since several are easy to confuse:
  calibrated     k_max = floor((n+1)*alpha - 1) >= 0, i.e. CRC has a miss budget
                 and some lambda > 0 exists. A class with k_max < 0 has no
                 operating point at that target.
  exceedance     test FNR > alpha on fold 10. CRC guarantees E[FNR] <= alpha in
                 expectation, so exceedances on a single fold are expected and
                 are not failures; the resampling column is the honest measure.
  resampling     fraction of bootstrap draws in which a class's TRUE FNR on the
  exceedance     pooled population exceeds alpha, averaged over calibrated
                 classes. This is the per-class analogue of the family-wise
                 rate, not a family-wise quantity itself.

Writes results/revision/e11_crc_configurations.csv. Alters no existing output.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.calibrate_acuity import load_tiers, ALPHA_MAPS
from src.revision.paths import run_dir, OUT
from src.revision.e2_per_class import clopper_pearson
from src.revision.e_crc import crc_threshold
from src.revision.e4_validity_sim import SEED as BOOT_SEED

N_TRIALS = 2000
CONFIGS = [("tiered", None), ("uniform_0.02", 0.02),
           ("uniform_0.05", 0.05), ("uniform_0.1", 0.10)]


def resampling_exceedance():
    """Mean over calibrated classes of P(true FNR > alpha), per configuration."""
    tiers, order = load_tiers(); col = {c: i for i, c in enumerate(order)}
    A = ALPHA_MAPS["primary"]
    d = run_dir("acuity35")
    s = np.concatenate([np.load(d/"scores_fold9.npy"),  np.load(d/"scores_fold10.npy")])
    y = np.concatenate([np.load(d/"labels_fold9.npy"),  np.load(d/"labels_fold10.npy")])
    pid = np.concatenate([
        np.array([r["patient_id"] for r in csv.DictReader(open(d/"meta_fold9.csv"))]),
        np.array([r["patient_id"] for r in csv.DictReader(open(d/"meta_fold10.csv"))])])
    n_cal = sum(1 for _ in csv.DictReader(open(d/"meta_fold9.csv")))
    tiered = [(c, t) for c, t in tiers if t != "Excluded"]

    pop = {c: np.sort(s[y[:, col[c]] == 1, col[c]]) for c, _ in tiered}
    tf = lambda c, l: float(np.searchsorted(pop[c], l, "left")/len(pop[c])) if len(pop[c]) else 0.0
    pats, inv = np.unique(pid, return_inverse=True)
    by = [np.nonzero(inv == i)[0] for i in range(len(pats))]
    rng = np.random.default_rng(BOOT_SEED)

    v = {k: {c: 0 for c, _ in tiered} for k, _ in CONFIGS}
    n = {k: {c: 0 for c, _ in tiered} for k, _ in CONFIGS}
    for _ in range(N_TRIALS):
        idx = np.concatenate([by[i] for i in rng.integers(0, len(pats), size=len(pats))])[:n_cal]
        cs, cy = s[idx], y[idx]
        for c, tier in tiered:
            j = col[c]
            pos = np.sort(cs[cy[:, j] == 1, j])
            for key, u in CONFIGS:
                a = A[tier] if u is None else u
                lam, k = crc_threshold(pos, a)
                if k >= 0:
                    n[key][c] += 1
                    if tf(c, lam) > a:
                        v[key][c] += 1
    return {k: {c: (v[k][c]/n[k][c] if n[k][c] else float("nan")) for c, _ in tiered}
            for k, _ in CONFIGS}


def main():
    rows = list(csv.DictReader(open(OUT/"e7_crc_per_class.csv")))
    ltt_cert = {r["class"] for r in csv.DictReader(open(OUT/"e2_per_class_seed0.csv"))
                if r["status"] == "certified"}
    print(f"  LTT-certified reference set: {len(ltt_cert)} classes")
    print("  running the bootstrap resampling arm ...")
    res = resampling_exceedance()

    out = []
    for key, u in CONFIGS:
        cal = [r for r in rows if int(r[f"{key}_k_max"]) >= 0]
        rec = {"config": "tiered (0.02/0.05/0.10)" if u is None else f"uniform {u:.2f}",
               "n_calibrated": len(cal)}
        for label, subset in (("all", cal),
                              ("ltt", [r for r in cal if r["class"] in ltt_cert])):
            n = len(subset)
            exc = sum(1 for r in subset
                      if float(r[f"{key}_test_fnr"]) > float(r[f"{key}_alpha"]))
            lo, hi = clopper_pearson(exc, n) if n else (float("nan"),)*2
            rec[f"{label}_n"] = n
            rec[f"{label}_exc"] = exc
            rec[f"{label}_rate"] = exc/n if n else float("nan")
            rec[f"{label}_lo"], rec[f"{label}_hi"] = lo, hi
            rec[f"{label}_flag"] = (float(np.mean([float(r[f"{key}_flag_rate"]) for r in subset]))
                                    if n else float("nan"))
        rec["resample"] = float(np.mean([res[key][r["class"]] for r in cal])) if cal else float("nan")
        # Critical-tier classes calibrated at a target looser than their tier's 0.02
        rec["crit_loose"] = ([r["class"] for r in cal if r["tier"] == "Critical"]
                             if (u is not None and u > 0.02) else [])
        out.append(rec)

    with open(OUT/"e11_crc_configurations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config","n_calibrated",
                    "all_n","all_exceedances","all_exceedance_rate","all_ci95_lo","all_ci95_hi",
                    "all_mean_flag_rate",
                    "ltt_n","ltt_exceedances","ltt_exceedance_rate","ltt_ci95_lo","ltt_ci95_hi",
                    "ltt_mean_flag_rate",
                    "mean_resampling_exceedance_rate",
                    "critical_classes_calibrated_at_looser_target"])
        for r in out:
            w.writerow([r["config"], r["n_calibrated"],
                        r["all_n"], r["all_exc"], round(r["all_rate"],6),
                        round(r["all_lo"],6), round(r["all_hi"],6), round(r["all_flag"],6),
                        r["ltt_n"], r["ltt_exc"], round(r["ltt_rate"],6),
                        round(r["ltt_lo"],6), round(r["ltt_hi"],6), round(r["ltt_flag"],6),
                        round(r["resample"],6), " ".join(sorted(r["crit_loose"]))])

    print(f"\n{'config':24s} {'cal':>4s} | {'all: exc':>9s} {'rate':>6s} {'95% CI':>16s} {'flag':>6s} "
          f"| {'LTT: exc':>9s} {'rate':>6s} {'flag':>6s} | {'resamp':>7s}")
    for r in out:
        print(f"{r['config']:24s} {r['n_calibrated']:>4d} | "
              f"{r['all_exc']:>4d}/{r['all_n']:<4d} {r['all_rate']:6.3f} "
              f"[{r['all_lo']:.3f},{r['all_hi']:.3f}] {r['all_flag']:6.3f} | "
              f"{r['ltt_exc']:>4d}/{r['ltt_n']:<4d} {r['ltt_rate']:6.3f} {r['ltt_flag']:6.3f} | "
              f"{r['resample']:7.3f}")
    print("\nCritical-tier classes calibrated at a target looser than their tier's 0.02:")
    for r in out:
        if r["crit_loose"]:
            print(f"  {r['config']:24s} {', '.join(sorted(r['crit_loose']))}")
    print("\nwrote e11_crc_configurations.csv")


if __name__ == "__main__":
    main()
