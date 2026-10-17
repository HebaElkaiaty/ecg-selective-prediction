"""Task 7 (E3): per-class Conformal Risk Control at uniform and tiered alpha.

Angelopoulos, Bates, Fisch, Lei & Schuster, "Conformal Risk Control" (2023).
For a loss bounded in [0, 1] and monotone in lambda, the CRC threshold is

    lambda_hat = sup{ lambda : (n * R_hat(lambda) + 1) / (n + 1) <= alpha }

which permits k_max = floor((n + 1) * alpha - 1) calibration misses. The
guarantee is on the EXPECTATION, E[FNR(lambda_hat)] <= alpha, over the draw of
the calibration set -- not a (1 - delta) high-probability statement, and it
carries no multiplicity correction. It is therefore weaker than the LTT
certificate but far more selective, because LTT at delta_k = 0.10/33 permits
zero misses for most classes.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.calibrate_acuity import load_tiers, ALPHA_MAPS, certify, build_candidates
from src.revision.e2_per_class import clopper_pearson, OUT

from src.revision.paths import run_dir
RUN = run_dir("acuity35")
DELTA = 0.10
UNIFORM = [0.02, 0.05, 0.10]


def crc_threshold(pos_scores: np.ndarray, alpha: float):
    """Return (lambda_hat, k_max). lambda = 0 when no miss budget exists."""
    n = len(pos_scores)
    if n == 0:
        return 0.0, -1
    k_max = int(np.floor((n + 1) * alpha - 1))
    if k_max < 0:
        return 0.0, k_max                      # budget too small for any lambda > 0
    p = np.sort(pos_scores)
    return (float(p[k_max]) if k_max < n else 1.0), k_max


def main():
    tiers, order = load_tiers(); col = {c: i for i, c in enumerate(order)}
    vs, vy = np.load(RUN/"scores_fold9.npy"),  np.load(RUN/"labels_fold9.npy")
    ts, ty = np.load(RUN/"scores_fold10.npy"), np.load(RUN/"labels_fold10.npy")
    tiered = [(c, t) for c, t in tiers if t != "Excluded"]
    dk = DELTA / len(tiered)
    A = ALPHA_MAPS["primary"]

    configs = [("tiered", None)] + [(f"uniform_{u}", u) for u in UNIFORM]
    out = []
    for c, tier in tiered:
        j = col[c]
        pos9 = vs[:, j][vy[:, j] == 1]
        pos10 = ty[:, j] == 1; n10 = int(pos10.sum())
        rec = dict(cls=c, tier=tier, n9=len(pos9), n10=n10)
        # LTT reference at the tiered target
        a_t = A[tier]
        lam_ltt = certify(pos9, a_t, dk, build_candidates("scores", pos9))
        rec["ltt_alpha"] = a_t; rec["ltt_lambda"] = lam_ltt
        rec["ltt_flag"] = float((ts[:, j] >= lam_ltt).mean())
        for name, u in configs:
            a = a_t if u is None else u
            lam, k = crc_threshold(pos9, a)
            pred = ts[:, j] >= lam
            miss = int((ts[pos10, j] < lam).sum())
            lo, hi = clopper_pearson(miss, n10)
            rec[name] = dict(alpha=a, k_max=k, lam=lam, miss=miss,
                             fnr=(miss/n10 if n10 else float("nan")),
                             lo=lo, hi=hi, flag=float(pred.mean()))
        out.append(rec)

    ORD = {"Critical":0,"Important":1,"Benign":2}
    out.sort(key=lambda r:(ORD[r["tier"]], r["cls"]))
    with open(OUT/"e7_crc_per_class.csv","w",newline="") as f:
        w=csv.writer(f)
        hdr=["class","tier","n_pos_fold9","n_pos_fold10","ltt_alpha","ltt_lambda","ltt_flag_rate"]
        for name,_ in configs:
            hdr += [f"{name}_{x}" for x in ("alpha","k_max","lambda","misses","test_fnr",
                                            "fnr_ci95_lo","fnr_ci95_hi","flag_rate")]
        w.writerow(hdr)
        for r in out:
            row=[r["cls"],r["tier"],r["n9"],r["n10"],r["ltt_alpha"],round(r["ltt_lambda"],6),
                 round(r["ltt_flag"],6)]
            for name,_ in configs:
                d=r[name]; row += [d["alpha"], d["k_max"], round(d["lam"],6), d["miss"],
                                   round(d["fnr"],6), round(d["lo"],6), round(d["hi"],6),
                                   round(d["flag"],6)]
            w.writerow(row)

    print(f"{'config':14s} {'lam>0':>6s} {'mean flag':>10s} {'mean FNR':>9s} {'over alpha':>11s}")
    for name,_ in configs:
        v=[r[name] for r in out]; nz=[d for d in v if d["lam"]>0]
        over=sum(1 for d in nz if d["fnr"]>d["alpha"])
        print(f"{name:14s} {len(nz):>6d} {np.mean([d['flag'] for d in nz]):>10.3f} "
              f"{np.mean([d['fnr'] for d in nz]):>9.4f} {over:>7d}/{len(nz)}")
    print("\nLTT vs CRC at the tiered target, classes LTT certifies:")
    print(f"{'class':10s} {'a':>5s} {'LTT lam':>10s} {'LTT flag':>9s} | {'k_max':>5s} "
          f"{'CRC lam':>10s} {'CRC flag':>9s} {'CRC FNR':>8s}")
    for r in out:
        if r["ltt_lambda"] > 0:
            d=r["tiered"]
            print(f"{r['cls']:10s} {r['ltt_alpha']:5.2f} {r['ltt_lambda']:10.6f} "
                  f"{r['ltt_flag']:9.3f} | {d['k_max']:>5d} {d['lam']:10.6f} "
                  f"{d['flag']:9.3f} {d['fnr']:8.4f}")
    print("\nwrote e7_crc_per_class.csv")


if __name__ == "__main__":
    main()
