"""Task 4: Table 2 (tier-to-alpha sensitivity) and Table 6 (tiered vs uniform alpha).

33-class family a priori, delta_k = 0.10/33, exact search over fold-9 positive scores.
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
DELTA = 0.10
UNIFORM = [0.02, 0.05, 0.10]
ORD = {"Critical": 0, "Important": 1, "Benign": 2}


def calibrate(alpha_of, tiered, col, vs, vy, ts, ty, dk):
    out = {}
    for c, tier in tiered:
        j, a = col[c], alpha_of[c] if isinstance(alpha_of, dict) and c in alpha_of else alpha_of[tier]
        pos9 = vs[:, j][vy[:, j] == 1]; n9 = len(pos9)
        lam = certify(pos9, a, dk, build_candidates("scores", pos9))
        st = "certified" if lam > 0 else ("trivial" if (1-a)**n9 <= dk else "uncertifiable")
        pred = ts[:, j] >= lam; pos = ty[:, j] == 1
        miss = int((~pred[pos]).sum()); n10 = int(pos.sum())
        out[c] = dict(tier=tier, alpha=a, n=n9, floor=floor_n(a, dk), lam=lam, status=st,
                      fnr=(miss/n10 if n10 else float("nan")), flag=float(pred.mean()),
                      miss=miss, n10=n10)
    return out


def main():
    tiers, order = load_tiers(); col = {c: i for i, c in enumerate(order)}
    vs, vy = np.load(RUN/"scores_fold9.npy"),  np.load(RUN/"labels_fold9.npy")
    ts, ty = np.load(RUN/"scores_fold10.npy"), np.load(RUN/"labels_fold10.npy")
    tiered = [(c, t) for c, t in tiers if t != "Excluded"]
    dk = DELTA/len(tiered)

    cfg = {k: calibrate(ALPHA_MAPS[k], tiered, col, vs, vy, ts, ty, dk)
           for k in ("stricter", "primary", "looser")}
    uni = {u: calibrate({t: u for t in ORD}, tiered, col, vs, vy, ts, ty, dk)
           for u in UNIFORM}

    # ---- Table 2: sensitivity to the tier-to-alpha mapping -------------------
    with open(OUT/"e4_table2_sensitivity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class","tier","n_pos_fold9"] + [f"{k}_{x}" for k in ("stricter","primary","looser")
                    for x in ("alpha","floor","status","lambda","test_fnr","flag_rate")])
        for c, t in sorted(tiered, key=lambda x: (ORD[x[1]], x[0])):
            row = [c, t, cfg["primary"][c]["n"]]
            for k in ("stricter","primary","looser"):
                d = cfg[k][c]
                row += [d["alpha"], d["floor"], d["status"], round(d["lam"],6),
                        round(d["fnr"],6), round(d["flag"],6)]
            w.writerow(row)
    for k in ("stricter","primary","looser"):
        n = sum(1 for d in cfg[k].values() if d["status"]=="certified")
        byt = {}
        for d in cfg[k].values():
            if d["status"]=="certified": byt[d["tier"]] = byt.get(d["tier"],0)+1
        fl = {t: floor_n(ALPHA_MAPS[k][t], dk) for t in ORD}
        print(f"  {k:9s} alpha={tuple(ALPHA_MAPS[k].values())} floors="
              f"{fl['Critical']}/{fl['Important']}/{fl['Benign']}  certified={n} "
              f"(Crit {byt.get('Critical',0)}, Imp {byt.get('Important',0)}, Ben {byt.get('Benign',0)})")

    # ---- Table 6: tiered vs uniform alpha ------------------------------------
    rows = []
    for c, t in sorted(tiered, key=lambda x: (ORD[x[1]], x[0])):
        tier_d = cfg["primary"][c]
        rec = dict(cls=c, tier=t, n=tier_d["n"], a_t=tier_d["alpha"],
                   t_status=tier_d["status"], t_lam=tier_d["lam"], t_fnr=tier_d["fnr"],
                   t_flag=tier_d["flag"])
        for u in UNIFORM:
            rec[f"u{u}"] = uni[u][c]["status"]
        # safety failure: certified under a uniform alpha LOOSER than the tier target
        rec["panel"] = "unaffected"
        looser_cert = [u for u in UNIFORM if u > rec["a_t"] and rec[f"u{u}"]=="certified"]
        if looser_cert:
            rec["panel"] = "safety_failure"
        elif rec["t_status"]=="certified" and any(rec[f"u{u}"]!="certified" for u in UNIFORM if u < rec["a_t"]):
            rec["panel"] = "efficiency_loss"
        rows.append(rec)

    with open(OUT/"e4_table6_uniform_vs_tiered.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["class","tier","n_pos_fold9","tiered_alpha","tiered_status","tiered_lambda",
                    "tiered_test_fnr","tiered_flag_rate"]+[f"uniform_{u}_status" for u in UNIFORM]+["panel"])
        for r in rows:
            w.writerow([r["cls"],r["tier"],r["n"],r["a_t"],r["t_status"],round(r["t_lam"],6),
                        round(r["t_fnr"],6),round(r["t_flag"],6)]+[r[f"u{u}"] for u in UNIFORM]+[r["panel"]])
    tally={}
    for r in rows: tally[r["panel"]]=tally.get(r["panel"],0)+1
    print("\n  Table 6 panels:", "  ".join(f"{k}={v}" for k,v in sorted(tally.items())))
    for u in UNIFORM:
        n=sum(1 for c,_ in tiered if uni[u][c]["status"]=="certified")
        byt={}
        for c,t in tiered:
            if uni[u][c]["status"]=="certified": byt[t]=byt.get(t,0)+1
        print(f"  uniform alpha={u:.2f}: certified={n} "
              f"(Crit {byt.get('Critical',0)}, Imp {byt.get('Important',0)}, Ben {byt.get('Benign',0)})")
    print("\nwrote e4_table2_sensitivity.csv, e4_table6_uniform_vs_tiered.csv")


if __name__ == "__main__":
    main()
