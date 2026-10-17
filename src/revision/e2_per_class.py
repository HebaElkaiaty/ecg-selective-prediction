"""E2: per-class test-fold operating characteristics with exact binomial intervals.

Locked configuration:
  folds 1-7 train / fold 8 select / fold 9 calibrate / fold 10 test
  Bonferroni family = all 33 non-excluded classes, delta_k = delta/33
  threshold candidates = observed fold-9 positive scores (exact search)
  seed 0 primary; seeds 0-4 report TRAINING variability (folds 9/10 are shared
  across seeds, so the seed spread does not probe the delta guarantee).

Outputs
  results/revision/e2_per_class_seed0.csv
  results/revision/e2_seed_variability.csv
  results/revision/e2_seed_auroc.csv
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.calibrate_acuity import certify, build_candidates, load_tiers, ALPHA_MAPS
from src.revision.paths import fmt_lambda

from src.revision.paths import CKPT_ROOT as R, OUT, SEEDS, run_dir
DELTA = 0.10


def clopper_pearson(k: int, n: int, conf: float = 0.95):
    """Exact two-sided binomial interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def floor_n(alpha: float, delta_k: float) -> int:
    """Smallest n+ for which H0 can be rejected at lambda=0 with zero misses.

    Exact integer search. The closed form ceil(log(delta_k)/log(1-alpha)) is
    off by one at small alpha (e.g. alpha=0.01 gives 577, but 0.99**577 still
    exceeds delta_k; the true floor is 578).
    """
    n = int(np.ceil(np.log(delta_k) / np.log(1.0 - alpha)))
    n = max(1, n - 2)
    while (1.0 - alpha) ** n > delta_k:
        n += 1
    return n


def run_seed(seed: int, alpha_map="primary", grid="scores", select="fold8"):
    tiers, order = load_tiers()
    col = {c: i for i, c in enumerate(order)}
    d = run_dir("acuity35", seed=seed, select=select)
    vs, vy = np.load(d / "scores_fold9.npy"),  np.load(d / "labels_fold9.npy")
    ts, ty = np.load(d / "scores_fold10.npy"), np.load(d / "labels_fold10.npy")

    tiered = [(c, t) for c, t in tiers if t != "Excluded"]
    delta_k = DELTA / len(tiered)                      # fixed33, a priori
    A = ALPHA_MAPS[alpha_map]

    rows = []
    for c, tier in tiered:
        j, a = col[c], A[tier]
        pos9 = vs[:, j][vy[:, j] == 1]
        n9 = len(pos9)
        lam = certify(pos9, a, delta_k, build_candidates(grid, pos9))
        certifiable = (1.0 - a) ** n9 <= delta_k
        status = "certified" if lam > 0 else ("trivial" if certifiable else "uncertifiable")

        pred = ts[:, j] >= lam
        pos = ty[:, j] == 1
        n10 = int(pos.sum())
        miss = int((~pred[pos]).sum())
        fnr = miss / n10 if n10 else float("nan")
        lo, hi = clopper_pearson(miss, n10)
        tn = int((~pred[~pos]).sum()); fp = int(pred[~pos].sum()); tp = int(pred[pos].sum())
        n_flag = int(pred.sum()); flo, fhi = clopper_pearson(n_flag, len(pred))
        rows.append(dict(
            cls=c, tier=tier, alpha=a, n_val_pos=n9, n_test_pos=n10,
            floor=floor_n(a, delta_k), status=status, lambda_c=lam,
            test_miss=miss, test_fnr=fnr, fnr_ci_lo=lo, fnr_ci_hi=hi,
            flag_rate=float(pred.mean()), flag_ci_lo=flo, flag_ci_hi=fhi,
            auto_ruleout_rate=float((~pred).mean()),
            specificity=(tn / (tn + fp)) if (tn + fp) else float("nan"),
            ppv=(tp / (tp + fp)) if (tp + fp) else float("nan"),
        ))
    auroc = {f"fold{f}": float(np.mean([
                roc_auc_score(y[:, k], s[:, k]) for k in range(y.shape[1])
                if len(np.unique(y[:, k])) >= 2]))
             for f, (s, y) in [(9, (vs, vy)), (10, (ts, ty))]}
    return rows, delta_k, auroc


def main():
    rows, delta_k, auroc0 = run_seed(0)
    cols = ["cls","tier","alpha","n_val_pos","n_test_pos","floor","status","lambda_c",
            "test_miss","test_fnr","fnr_ci_lo","fnr_ci_hi","flag_rate","flag_ci_lo",
            "flag_ci_hi","auto_ruleout_rate","specificity","ppv"]
    hdr  = ["class","tier","alpha","n_pos_fold9","n_pos_fold10","cert_floor","status",
            "lambda_c","test_misses","test_fnr","fnr_ci95_lo","fnr_ci95_hi","flag_rate",
            "flag_ci95_lo","flag_ci95_hi","auto_ruleout_rate","specificity","ppv"]
    o = {"Critical":0,"Important":1,"Benign":2}
    rows.sort(key=lambda r:(o[r["tier"]], r["cls"]))
    with open(OUT/"e2_per_class_seed0.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(hdr+["delta","delta_k","family_size"])
        for r in rows:
            w.writerow([fmt_lambda(r[c]) if c == "lambda_c"
                        else (r[c] if not isinstance(r[c], float) else round(r[c], 6))
                        for c in cols] + [DELTA, round(delta_k, 8), 33])
    cert = [r["cls"] for r in rows if r["status"]=="certified"]
    print(f"e2_per_class_seed0.csv -- 33 classes, delta_k={delta_k:.8f}, "
          f"{len(cert)} certified")

    # seed variability: TRAINING variability (folds 9/10 shared across seeds)
    per = {s: {r["cls"]: r for r in run_seed(s)[0]} for s in SEEDS}
    aur = {s: run_seed(s)[2] for s in SEEDS}
    with open(OUT/"e2_seed_variability.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["class","tier","alpha","n_seeds_certified",
                    "lambda_mean","lambda_sd","test_fnr_mean","test_fnr_sd",
                    "flag_rate_mean","flag_rate_sd"])
        for c in cert:
            v=[per[s][c] for s in SEEDS if per[s][c]["status"]=="certified"]
            g=lambda k: np.array([x[k] for x in v], float)
            w.writerow([c, per[0][c]["tier"], per[0][c]["alpha"], len(v)]
                       + [round(float(x),6) for x in
                          (g("lambda_c").mean(), g("lambda_c").std(ddof=1),
                           g("test_fnr").mean(), g("test_fnr").std(ddof=1),
                           g("flag_rate").mean(), g("flag_rate").std(ddof=1))])
    with open(OUT/"e2_seed_auroc.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["seed","macro_auroc_fold9","macro_auroc_fold10","n_certified"])
        for s in SEEDS:
            w.writerow([s, round(aur[s]["fold9"],6), round(aur[s]["fold10"],6),
                        sum(1 for r in per[s].values() if r["status"]=="certified")])
    print("e2_seed_variability.csv, e2_seed_auroc.csv written")


if __name__ == "__main__":
    main()
