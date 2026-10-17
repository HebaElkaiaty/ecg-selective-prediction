"""Task 4 (E5): tier reassignment variants.  Task 5 (E7): delta = 0.05.

Family remains all 33 non-excluded classes; exact search; seed-0 fold8 model.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.calibrate_acuity import certify, build_candidates, load_tiers, ALPHA_MAPS
from src.revision.e2_per_class import floor_n, clopper_pearson, OUT

from src.revision.paths import run_dir
RUN = run_dir("acuity35")
A = ALPHA_MAPS["primary"]
ORD = {"Critical": 0, "Important": 1, "Benign": 2}
VARIANTS = {
    "baseline":            {},
    "AFIB_AFLT_Important": {"AFIB": "Important", "AFLT": "Important"},
    "AVB_Important":       {"_AVB": "Important"},
    "LMI_PMI_Important":   {"LMI": "Important", "PMI": "Important"},
}


def run(tier_of, delta, tiered, col, vs, vy, ts, ty):
    dk = delta/len(tiered)
    out = {}
    for c, _ in tiered:
        t = tier_of[c]; a = A[t]; j = col[c]
        pos9 = vs[:, j][vy[:, j] == 1]; n9 = len(pos9)
        lam = certify(pos9, a, dk, build_candidates("scores", pos9))
        st = "certified" if lam > 0 else ("trivial" if (1-a)**n9 <= dk else "uncertifiable")
        pred = ts[:, j] >= lam; pos = ty[:, j] == 1
        miss = int((~pred[pos]).sum()); n10 = int(pos.sum())
        lo, hi = clopper_pearson(miss, n10)
        out[c] = dict(tier=t, alpha=a, n9=n9, floor=floor_n(a, dk), lam=lam, status=st,
                      fnr=(miss/n10 if n10 else float("nan")), lo=lo, hi=hi,
                      flag=float(pred.mean()), miss=miss, n10=n10)
    return out, dk


def main():
    tiers, order = load_tiers(); col = {c: i for i, c in enumerate(order)}
    base_tier = {c: t for c, t in tiers}
    vs, vy = np.load(RUN/"scores_fold9.npy"),  np.load(RUN/"labels_fold9.npy")
    ts, ty = np.load(RUN/"scores_fold10.npy"), np.load(RUN/"labels_fold10.npy")
    tiered = [(c, t) for c, t in tiers if t != "Excluded"]

    # ---- Task 4: tier reassignment -----------------------------------------
    rows = []
    for name, ov in VARIANTS.items():
        tier_of = dict(base_tier); tier_of.update(ov)
        res, dk = run(tier_of, 0.10, tiered, col, vs, vy, ts, ty)
        cert = [c for c in res if res[c]["status"] == "certified"]
        byt = {}
        for c in cert: byt[res[c]["tier"]] = byt.get(res[c]["tier"], 0) + 1
        mf = float(np.mean([res[c]["flag"] for c in cert]))
        print(f"  {name:22s} certified={len(cert):2d} "
              f"(Crit {byt.get('Critical',0)}, Imp {byt.get('Important',0)}, "
              f"Ben {byt.get('Benign',0)})  mean flag {mf:.3f}")
        moved = sorted(ov)
        for c in moved:
            d = res[c]; b = run(base_tier, 0.10, tiered, col, vs, vy, ts, ty)[0][c]
            print(f"      {c}: {base_tier[c]} -> {d['tier']}  "
                  f"status {b['status']} -> {d['status']}  "
                  f"floor {b['floor']} -> {d['floor']}  (n={d['n9']})")
        for c in sorted(res, key=lambda x: (ORD[res[x]["tier"]], x)):
            d = res[c]
            rows.append([name, c, d["tier"], d["alpha"], d["n9"], d["floor"], d["status"],
                         round(d["lam"],6), round(d["fnr"],6), round(d["lo"],6),
                         round(d["hi"],6), round(d["flag"],6),
                         "yes" if c in ov else ""])
    with open(OUT/"e5_tier_reassignment.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["variant","class","tier","alpha","n_pos_fold9","cert_floor","status",
                    "lambda_c","test_fnr","fnr_ci95_lo","fnr_ci95_hi","flag_rate","reassigned"])
        w.writerows(rows)

    # ---- Task 5: delta = 0.05 ----------------------------------------------
    print("\n  delta = 0.05:")
    dk05 = 0.05/len(tiered)
    print(f"    delta_k = {dk05:.8f}  floors "
          f"{floor_n(0.02,dk05)}/{floor_n(0.05,dk05)}/{floor_n(0.10,dk05)}")
    r10,_ = run(base_tier, 0.10, tiered, col, vs, vy, ts, ty)
    r05,_ = run(base_tier, 0.05, tiered, col, vs, vy, ts, ty)
    c10 = {c for c in r10 if r10[c]["status"]=="certified"}
    c05 = {c for c in r05 if r05[c]["status"]=="certified"}
    byt={}
    for c in c05: byt[r05[c]["tier"]]=byt.get(r05[c]["tier"],0)+1
    print(f"    certified {len(c10)} (delta=0.10) -> {len(c05)} (delta=0.05)  "
          f"(Crit {byt.get('Critical',0)}, Imp {byt.get('Important',0)}, Ben {byt.get('Benign',0)})")
    for c in sorted(c10 - c05):
        print(f"      LOST {c:10s} tier {r10[c]['tier']:9s} n={r10[c]['n9']:3d} "
              f"floor {r10[c]['floor']} -> {r05[c]['floor']}")
    for c in sorted(c05 - c10):
        print(f"      GAINED {c}")
    with open(OUT/"e7_delta005.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["class","tier","alpha","n_pos_fold9","floor_delta010","status_delta010",
                    "lambda_delta010","floor_delta005","status_delta005","lambda_delta005",
                    "test_fnr_delta005","flag_rate_delta005"])
        for c in sorted(r05, key=lambda x:(ORD[r05[x]["tier"]], x)):
            w.writerow([c, r05[c]["tier"], r05[c]["alpha"], r05[c]["n9"],
                        r10[c]["floor"], r10[c]["status"], round(r10[c]["lam"],6),
                        r05[c]["floor"], r05[c]["status"], round(r05[c]["lam"],6),
                        round(r05[c]["fnr"],6), round(r05[c]["flag"],6)])
    print("\nwrote e5_tier_reassignment.csv, e7_delta005.csv")


if __name__ == "__main__":
    main()
