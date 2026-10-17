"""R3 Q2: multiplicity corrections, and the sample-size requirement.

TWO PARTS, AND ONLY ONE OF THEM IS A RESULT.

Part 1 -- Holm and BH. DO NOT QUOTE THESE COUNTS AS CERTIFICATIONS.
The comparison below models each class as a SINGLE hypothesis, scored by its
best achievable p-value (1-alpha)^n. That is not the procedure. LTT tests a
chain of thresholds per class, and the selected lambda is chosen precisely
because its p-value clears the level, so the p-value is not fixed independently
of the level the way Holm's step-down requires. The chain's terminating step is
a non-rejection by construction, so no error budget is ever freed for
redistribution to another class. The apparent extra certification (CRBBB) is an
artefact of the single-test model, not a class the procedure can certify.

Bonferroni is the correction used, and the only one. These numbers exist to
support one sentence in the R3 Q2 reply -- that Holm offers no gain here for
the reason above, and that BH controls the false discovery rate, which is not
the simultaneous statement the paper makes -- and for nothing else.

Part 2 -- the sample-size requirement. This IS a result, and answers the other
half of R3 Q2: how large a calibration set each uncertifiable class would need.
n+ >= log(delta_k)/log(1-alpha), which no amount of model improvement lowers.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.calibrate_acuity import certify, build_candidates, load_tiers, ALPHA_MAPS
from src.revision.paths import run_dir, OUT

DELTA = 0.10
ORD = {"Critical": 0, "Important": 1, "Benign": 2}


def floor_for(alpha, level):
    n = 1
    while (1.0 - alpha) ** n > level:
        n += 1
    return n


def main():
    tiers, order = load_tiers(); col = {c: i for i, c in enumerate(order)}
    A = ALPHA_MAPS["primary"]
    d = run_dir("acuity35")
    vs, vy = np.load(d/"scores_fold9.npy"), np.load(d/"labels_fold9.npy")
    ts, ty = np.load(d/"scores_fold10.npy"), np.load(d/"labels_fold10.npy")
    tiered = [(c, t) for c, t in tiers if t != "Excluded"]
    m = len(tiered)

    info = {}
    for c, tier in tiered:
        j, a = col[c], A[tier]
        pos = vs[:, j][vy[:, j] == 1]; n = len(pos)
        info[c] = dict(tier=tier, alpha=a, n=n, pmin=float((1.0 - a) ** n), pos=pos, j=j)

    ranked = sorted(tiered, key=lambda x: info[x[0]]["pmin"])

    # Bonferroni
    bonf_level = {c: DELTA / m for c, _ in tiered}
    # Holm step-down
    holm_level, stopped = {}, False
    for i, (c, _) in enumerate(ranked, start=1):
        lvl = DELTA / (m - i + 1)
        holm_level[c] = lvl
        if not stopped and info[c]["pmin"] > lvl:
            stopped = True
        info[c]["holm_rejected"] = (not stopped)
    # BH step-up
    bh_level = {c: (i / m) * DELTA for i, (c, _) in enumerate(ranked, start=1)}
    kmax = 0
    for i, (c, _) in enumerate(ranked, start=1):
        if info[c]["pmin"] <= bh_level[c]:
            kmax = i
    for i, (c, _) in enumerate(ranked, start=1):
        info[c]["bh_rejected"] = (i <= kmax)

    rows = []
    for c, tier in tiered:
        d_ = info[c]
        rec = dict(cls=c, tier=tier, alpha=d_["alpha"], n=d_["n"], pmin=d_["pmin"])
        for name, lvl, rej in [
            ("bonferroni", bonf_level[c], d_["pmin"] <= bonf_level[c]),
            ("holm",       holm_level[c], d_["holm_rejected"]),
            ("bh",         bh_level[c],   d_["bh_rejected"])]:
            lam = certify(d_["pos"], d_["alpha"], lvl,
                          build_candidates("scores", d_["pos"])) if rej else 0.0
            pred = ts[:, d_["j"]] >= lam; pos10 = ty[:, d_["j"]] == 1
            rec[name] = dict(level=lvl, cert=bool(lam > 0), lam=lam,
                             floor=floor_for(d_["alpha"], lvl),
                             fnr=(float((~pred[pos10]).mean()) if pos10.sum() else float("nan")),
                             flag=float(pred.mean()))
        rows.append(rec)

    rows.sort(key=lambda r: (ORD[r["tier"]], r["cls"]))
    with open(OUT/"e9_multiplicity_corrections.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class","tier","alpha","n_pos_fold9","min_achievable_p"] +
                   [f"{k}_{x}" for k in ("bonferroni","holm","bh")
                    # "would_reject", not "certified": only Bonferroni's column is a
                    # certification. See the module docstring.
                    for x in ("level","floor","would_reject","lambda","test_fnr","flag_rate")])
        for r in rows:
            out = [r["cls"], r["tier"], r["alpha"], r["n"], round(r["pmin"], 8)]
            for k in ("bonferroni", "holm", "bh"):
                d_ = r[k]
                out += [round(d_["level"], 8), d_["floor"], "yes" if d_["cert"] else "no",
                        round(d_["lam"], 6), round(d_["fnr"], 6), round(d_["flag"], 6)]
            w.writerow(out)

    print("  NOTE: Holm/BH rows below are a single-test-per-class model, NOT the")
    print("  procedure. They answer R3 Q2 in one sentence; they are not certifications.\n")
    print(f"{'correction':12s} {'would reject':>12s}  {'Crit':>4s} {'Imp':>4s} {'Ben':>4s}   difference vs Bonferroni")
    base = {r["cls"] for r in rows if r["bonferroni"]["cert"]}
    for k, lbl in [("bonferroni","Bonferroni"),("holm","Holm"),("bh","BH (FDR)")]:
        cert = {r["cls"] for r in rows if r[k]["cert"]}
        byt = {}
        for r in rows:
            if r[k]["cert"]: byt[r["tier"]] = byt.get(r["tier"], 0) + 1
        gained = sorted(cert - base)
        print(f"{lbl:12s} {len(cert):>10d}  {byt.get('Critical',0):>4d} {byt.get('Important',0):>4d} "
              f"{byt.get('Benign',0):>4d}   {', '.join(gained) if gained else '-'}")

    # Sample-size requirement for the classes that cannot be certified
    print(f"\nCalibration positives required (33-class family, delta_k = {DELTA}/{m}):")
    print(f"{'class':10s} {'tier':10s} {'alpha':>5s} {'have':>5s} {'need(0 miss)':>13s} "
          f"{'need(2 miss)':>13s} {'x fold 9':>9s}")
    from scipy import stats
    need_rows = []
    for r in rows:
        if r["bonferroni"]["cert"]: continue
        a, lvl = r["alpha"], DELTA/m
        n0 = floor_for(a, lvl)
        n2 = next(n for n in range(1, 20000) if stats.binom.cdf(2, n, a) <= lvl)
        prev = r["n"] / 2183.0
        need_rows.append((r["cls"], r["tier"], a, r["n"], n0, n2, n0/r["n"] if r["n"] else float("inf")))
        print(f"{r['cls']:10s} {r['tier']:10s} {a:5.2f} {r['n']:>5d} {n0:>13d} {n2:>13d} "
              f"{(n0/prev)/2183.0 if prev else float('inf'):>9.1f}")
    with open(OUT/"e9_sample_size_requirement.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class","tier","alpha","n_pos_fold9","n_pos_needed_zero_miss",
                    "n_pos_needed_two_misses","calibration_records_needed_zero_miss",
                    "multiple_of_fold9"])
        for c, t, a, n, n0, n2, _ in need_rows:
            prev = n/2183.0
            recs = n0/prev if prev else float("nan")
            w.writerow([c, t, a, n, n0, n2, int(round(recs)) if prev else "",
                        round(recs/2183.0, 1) if prev else ""])
    print("\nwrote e9_multiplicity_corrections.csv, e9_sample_size_requirement.csv")


if __name__ == "__main__":
    main()
