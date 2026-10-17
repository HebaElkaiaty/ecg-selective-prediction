"""Acuity-weighted class-conditional risk control -- calibration driver.

Regenerates the per-class LTT thresholds from saved model outputs, with the
class family and the threshold candidate set as explicit, auditable choices
rather than implicit ones.

Class family (--family)
-----------------------
legacy24  Flat n_val+ >= 10, as actually used in the submitted run. |Y_sup| = 24,
          delta_k = delta/24. Reproduces the submitted primary_calibration.csv.

fixed33   All 33 non-excluded classes, fixed a priori before looking at the data.
          |Y_sup| = 33, delta_k = delta/33. This removes two objections at once:
          the submitted Table 1 caption claims a tier-specific floor
          ceil(1/alpha)-1 that the code did not implement (R1, R4.6d), and the
          flat rule makes the number of hypotheses data-dependent, since it is
          derived from calibration-fold counts.

Status
------
uncertifiable  (1-alpha)^n_pos > delta_k -- H0 cannot be rejected even at lambda=0
               with zero observed misses. Depends only on n_pos, alpha, delta_k.
trivial        Clears that floor, but no lambda > 0 passes the binomial test.
               Reported lambda = 0: flags every record, FNR = 0 by convention.
certified      lambda_c > 0.

Tier -> alpha comes from src/calibration/acuity_tiers.csv, the single source of
truth for the class/tier/alpha assignment. Never hardcode a second copy.

Usage
-----
    # new run
    python -m src.calibrate_acuity --source checkpoints/retrain/acuity35_seed0_fold8 \
        --family fixed33 --out results/revision/calibration_seed0_fixed33.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration.risk_control import calibrate_threshold
from src.revision.paths import fmt_lambda

TIERS_CSV = Path(__file__).resolve().parent / "calibration" / "acuity_tiers.csv"
GRID_SIZE = 1000                      # np.linspace(0, 1, 1000)
LEGACY_FLAT_FLOOR = 10                # the flat rule the submitted run used

# Sensitivity configurations (Section 3.3.4)
ALPHA_MAPS = {
    "primary":  {"Critical": 0.02, "Important": 0.05, "Benign": 0.10},
    "stricter": {"Critical": 0.01, "Important": 0.03, "Benign": 0.08},
    "looser":   {"Critical": 0.05, "Important": 0.08, "Benign": 0.15},
}


def certify(pos_scores: np.ndarray, alpha: float, delta_k: float,
             candidates: np.ndarray) -> float:
    """Largest candidate threshold rejecting H0: FNR >= alpha, at level delta_k.

    Identical binomial math to risk_control.calibrate_threshold -- same
    stats.binom.cdf test, same monotone-boundary search. The only difference is
    that the candidate set is supplied explicitly rather than fixed to
    linspace(0, 1, 1000). Validated against that function on the uniform grid.

    Grid size carries no multiplicity cost here: FNR(lambda) is non-decreasing,
    so the rejection region is a lower interval and only its boundary is tested
    (bonferroni_grid=False). Refining the candidate set is therefore free.
    """
    n = len(pos_scores)
    if n == 0:
        return 0.0
    cand = np.sort(np.asarray(candidates, dtype=float))
    k = np.searchsorted(np.sort(pos_scores), cand, side="left")   # misses at each lambda
    p = stats.binom.cdf(k, n, alpha)
    cut = int(np.searchsorted(p, delta_k, side="right")) - 1
    return float(cand[cut]) if cut >= 0 else 0.0


def build_candidates(kind: str, pos_scores: np.ndarray) -> np.ndarray:
    """Candidate thresholds."""
    if kind == "uniform":
        return np.linspace(0.0, 1.0, GRID_SIZE)
    if kind == "scores":
        # Empirical FNR is a step function changing only at positive-score
        # values, so these candidates are exhaustive, not an approximation.
        return np.unique(np.concatenate([[0.0], np.asarray(pos_scores, float)]))
    if kind == "log":
        return np.unique(np.concatenate([[0.0], np.logspace(-6, 0, GRID_SIZE)]))
    raise ValueError(kind)


def load_tiers() -> list[tuple[str, str]]:
    """Return [(class, tier)] in model column order, excluding NORM/SR."""
    rows = list(csv.DictReader(open(TIERS_CSV)))
    return [(r["class"], r["tier"]) for r in rows], [r["class"] for r in rows]


def load_outputs(source: str):
    """Return (val_scores, val_labels, test_scores, test_labels)."""
    if source == "legacy":
        ck = Path("checkpoints")
        sig = lambda x: 1.0 / (1.0 + np.exp(-x))
        return (sig(np.load(ck / "acuity_val_logits.npy")),
                np.load(ck / "acuity_val_labels.npy"),
                sig(np.load(ck / "acuity_test_logits.npy")),
                np.load(ck / "acuity_test_labels.npy"))
    d = Path(source)
    return (np.load(d / "scores_fold9.npy"),  np.load(d / "labels_fold9.npy"),
            np.load(d / "scores_fold10.npy"), np.load(d / "labels_fold10.npy"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="'legacy' for the original checkpoints, or a "
                         "checkpoints/retrain/<run> directory")
    ap.add_argument("--family", choices=["legacy24", "fixed33"], required=True)
    ap.add_argument("--alpha-map", choices=list(ALPHA_MAPS), default="primary")
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--method", default="binomial", choices=["binomial", "hoeffding"])
    ap.add_argument("--grid", choices=["uniform", "scores", "log"], default="uniform",
                    help="candidate thresholds. 'uniform' = linspace(0,1,1000), the "
                         "submitted setting. 'scores' = observed positive scores "
                         "(exhaustive; no multiplicity cost under monotone search).")
    ap.add_argument("--out", default=None, help="output CSV path")
    ap.add_argument("--validate-against", default=None,
                    help="existing primary_calibration.csv to diff against")
    args = ap.parse_args()

    tiers, col_order = load_tiers()
    alpha_of = ALPHA_MAPS[args.alpha_map]
    col = {c: i for i, c in enumerate(col_order)}

    vs, vy, ts, ty = load_outputs(args.source)
    assert vs.shape[1] == len(col_order) == ts.shape[1], "class-count mismatch"

    tiered = [(c, t) for c, t in tiers if t != "Excluded"]
    n_pos = {c: int(vy[:, col[c]].sum()) for c, _ in tiered}

    if args.family == "legacy24":
        family = [c for c, _ in tiered if n_pos[c] >= LEGACY_FLAT_FLOOR]
    else:
        family = [c for c, _ in tiered]

    delta_k = args.delta / len(family)
    print(f"source={args.source}  family={args.family} (|Y_sup|={len(family)})  "
          f"alpha-map={args.alpha_map}  grid={args.grid}  "
          f"delta={args.delta}  delta_k={delta_k:.8f}")

    rows = []
    for c, tier in tiered:
        j, a, n = col[c], alpha_of[tier], n_pos[c]
        if c not in family:
            rows.append(dict(cls=c, tier=tier, alpha=a, n_val_pos=n, lambda_c="",
                             test_fnr="", flag_pct="", status="not_in_family"))
            continue

        pos_scores = vs[:, j][vy[:, j] == 1]
        if args.grid == "uniform" and args.method == "binomial":
            lam = calibrate_threshold(vs[:, j], vy[:, j], alpha=a, delta=delta_k,
                                      grid_size=GRID_SIZE, bonferroni_grid=False,
                                      method=args.method)
        else:
            lam = certify(pos_scores, a, delta_k, build_candidates(args.grid, pos_scores))
        # Floor test: can H0 be rejected at all, at lambda=0 with zero misses?
        certifiable = (1.0 - a) ** n <= delta_k
        status = "certified" if lam > 0 else ("trivial" if certifiable else "uncertifiable")

        pred = ts[:, j] >= lam
        pos = ty[:, j] == 1
        test_fnr = float((~pred[pos]).mean()) if pos.sum() else 0.0
        rows.append(dict(cls=c, tier=tier, alpha=a, n_val_pos=n,
                         lambda_c=fmt_lambda(lam), test_fnr=round(test_fnr, 6),
                         flag_pct=round(100.0 * pred.mean(), 2), status=status,
                         n_test_pos=int(pos.sum()), n_test_miss=int((~pred[pos]).sum())))

    order = {"Critical": 0, "Important": 1, "Benign": 2}
    rows.sort(key=lambda r: (order[r["tier"]], r["cls"]))

    out = Path(args.out) if args.out else Path(
        f"results/revision/calibration_{args.family}_{args.alpha_map}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "tier", "alpha", "n_val_pos", "lambda_c", "test_fnr",
                    "flag_pct", "status", "n_test_pos", "n_test_miss",
                    "delta", "delta_k", "family_size", "method"])
        for r in rows:
            w.writerow([r["cls"], r["tier"], r["alpha"], r["n_val_pos"], r["lambda_c"],
                        r["test_fnr"], r["flag_pct"], r["status"],
                        r.get("n_test_pos", ""), r.get("n_test_miss", ""),
                        args.delta, round(delta_k, 8), len(family), args.method])

    tally = {}
    for r in rows:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    print(f"wrote {out}")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))

    if args.validate_against:
        ref = {r["class"]: r for r in csv.DictReader(open(args.validate_against))}
        got = {r["cls"]: r for r in rows}
        diffs, checked = [], 0
        for c, r in ref.items():
            g = got.get(c)
            if g is None:
                diffs.append(f"{c}: missing from new output"); continue
            checked += 1
            for key, rk in [("lambda_c", "lambda_c"), ("test_fnr", "test_fnr"),
                            ("flag_pct", "flag_pct")]:
                if abs(float(r[rk]) - float(g[key] or 0)) > 5e-4:
                    diffs.append(f"{c}.{key}: ref={r[rk]} new={g[key]}")
            if r["status"] != g["status"]:
                diffs.append(f"{c}.status: ref={r['status']} new={g['status']}")
        print(f"\nVALIDATION vs {args.validate_against}  ({checked} classes)")
        print("  EXACT MATCH" if not diffs else "  MISMATCHES:")
        for d in diffs:
            print(f"    {d}")


if __name__ == "__main__":
    main()
