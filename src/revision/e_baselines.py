"""Task 3: uncorrected baselines (temperature scaling, MC dropout) vs LTT.

Denominator is the certified set only (12 classes): a class with no operating
point cannot violate a target, so the submitted 0/24 denominator was not
well defined.

Baseline rule (unchanged from the submitted run): per class, the largest lambda
whose empirical fold-9 FNR is <= alpha_tier, with no allowance for sampling
variability between calibration and test folds.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.calibrate_acuity import certify, build_candidates, load_tiers, ALPHA_MAPS
from src.metrics import temperature_scale, expected_calibration_error
from src.revision.e2_per_class import clopper_pearson, OUT

from src.revision.paths import run_dir
RUN = run_dir("acuity35")
DELTA, N_BINS = 0.10, 15


def naive_threshold(sv, lv, alpha):
    """Largest lambda with empirical val FNR <= alpha. No correction."""
    pos = np.sort(sv[lv == 1]); n = len(pos)
    if n == 0:
        return 0.0
    k = int(np.floor(alpha * n))
    return 1.0 if k >= n else float(pos[k])


def evaluate(st, lt, lam):
    pos = lt == 1; n = int(pos.sum())
    miss = int((st[pos] < lam).sum())
    lo, hi = clopper_pearson(miss, n)
    return dict(lam=lam, miss=miss, n=n, fnr=(miss / n if n else float("nan")),
                lo=lo, hi=hi, flag=float((st >= lam).mean()))


def main():
    tiers, order = load_tiers(); col = {c: i for i, c in enumerate(order)}
    A = ALPHA_MAPS["primary"]
    vl = np.load(RUN/"logits_fold9.npy");  vy = np.load(RUN/"labels_fold9.npy")
    tl = np.load(RUN/"logits_fold10.npy"); ty = np.load(RUN/"labels_fold10.npy")
    vs, ts = expit(vl), expit(tl)
    vmc = np.load(RUN/"mc_probs_fold9.npy"); tmc = np.load(RUN/"mc_probs_fold10.npy")

    T = temperature_scale(vl, vy)
    vts, tts = expit(vl / T), expit(tl / T)
    ece = {"val_before": expected_calibration_error(vs, vy, N_BINS),
           "val_after":  expected_calibration_error(vts, vy, N_BINS),
           "test_before": expected_calibration_error(ts, ty, N_BINS),
           "test_after":  expected_calibration_error(tts, ty, N_BINS)}
    print(f"Temperature T = {T:.4f}   (ECE pooled over all sample-class pairs, {N_BINS} bins)")
    for k, v in ece.items():
        print(f"  ECE {k:12s} {v:.6f}")
    with open(OUT/"e3_temperature_ece.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["quantity","value","n_bins","pooling","fitted_on"])
        w.writerow(["temperature_T", round(T,6), "", "", "fold9"])
        for k,v in ece.items():
            w.writerow([f"ece_{k}", round(v,6), N_BINS, "all sample-class pairs", "fold9"])

    tiered = [(c,t) for c,t in tiers if t!="Excluded"]
    dk = DELTA/len(tiered)
    rows, viol = [], {"LTT":0,"TS":0,"MC":0}
    for c,tier in tiered:
        j,a = col[c], A[tier]
        pos9 = vs[:,j][vy[:,j]==1]
        lam = certify(pos9, a, dk, build_candidates("scores", pos9))
        if lam <= 0:
            continue                                   # not certified: no operating point
        r = {"LTT": evaluate(ts[:,j], ty[:,j], lam),
             "TS":  evaluate(tts[:,j], ty[:,j], naive_threshold(vts[:,j], vy[:,j], a)),
             "MC":  evaluate(tmc[:,j], ty[:,j], naive_threshold(vmc[:,j], vy[:,j], a))}
        for m in r:
            if r[m]["fnr"] > a: viol[m]+=1
        rows.append((c,tier,a,r))

    n = len(rows)
    with open(OUT/"e3_baseline_per_class.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["class","tier","alpha","n_test_pos"]+
                   [f"{m}_{k}" for m in ("ltt","ts","mc")
                    for k in ("lambda","misses","test_fnr","fnr_ci95_lo","fnr_ci95_hi","flag_rate","violates")])
        for c,tier,a,r in rows:
            out=[c,tier,a,r["LTT"]["n"]]
            for m in ("LTT","TS","MC"):
                d=r[m]; out += [round(d["lam"],6), d["miss"], round(d["fnr"],6),
                                round(d["lo"],6), round(d["hi"],6), round(d["flag"],6),
                                "yes" if d["fnr"]>a else ""]
            w.writerow(out)

    with open(OUT/"e3_baseline_headline.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["method","guarantee_type","violations","denominator",
                                     "rate","rate_ci95_lo","rate_ci95_hi","mean_flag_rate"])
        for m,g in [("LTT binomial","Statistical (binomial LTT)"),
                    ("Temperature scaling (uncorrected)","Empirical"),
                    ("MC Dropout (uncorrected)","Empirical")]:
            key={"LTT binomial":"LTT","Temperature scaling (uncorrected)":"TS",
                 "MC Dropout (uncorrected)":"MC"}[m]
            v=viol[key]; lo,hi=clopper_pearson(v,n)
            mf=float(np.mean([r[key]["flag"] for _,_,_,r in rows]))
            w.writerow([m,g,v,n,round(v/n,4),round(lo,4),round(hi,4),round(mf,4)])
            print(f"  {m:36s} {v}/{n}  rate {v/n:.3f} CI[{lo:.3f},{hi:.3f}]  mean flag {mf:.3f}")
    print(f"\nwrote e3_baseline_per_class.csv, e3_baseline_headline.csv (denominator = {n} certified)")


if __name__ == "__main__":
    main()
