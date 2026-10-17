"""Supplementary: (a) final-epoch model, same calibration; (b) uniform grid vs exact search."""
from __future__ import annotations
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.revision.e2_per_class import run_seed, OUT
from src.revision.paths import fmt_lambda

COLS = ["cls","tier","alpha","n_val_pos","n_test_pos","floor","status","lambda_c",
        "test_miss","test_fnr","fnr_ci_lo","fnr_ci_hi","flag_rate","flag_ci_lo",
        "flag_ci_hi","auto_ruleout_rate","specificity","ppv"]
HDR  = ["class","tier","alpha","n_pos_fold9","n_pos_fold10","cert_floor","status",
        "lambda_c","test_misses","test_fnr","fnr_ci95_lo","fnr_ci95_hi","flag_rate",
        "flag_ci95_lo","flag_ci95_hi","auto_ruleout_rate","specificity","ppv"]
ORD = {"Critical":0,"Important":1,"Benign":2}

def write(rows, path, extra_hdr=(), extra=()):
    rows = sorted(rows, key=lambda r:(ORD[r["tier"]], r["cls"]))
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(HDR+list(extra_hdr))
        for r in rows:
            w.writerow([fmt_lambda(r[c]) if c == "lambda_c"
                        else (round(r[c], 6) if isinstance(r[c], float) else r[c])
                        for c in COLS] + list(extra))
    return rows

# (a) seed-0 final-epoch checkpoint, identical calibration
fin,_,au = run_seed(0, grid="scores", select="final")
write(fin, OUT/"supp_a_final_epoch_seed0.csv", ["checkpoint_rule"], ["final_epoch"])
nf = sum(1 for r in fin if r["status"]=="certified")
print(f"(a) final-epoch seed 0: {nf} certified | fold9 AUROC {au['fold9']:.4f} fold10 {au['fold10']:.4f}")

# (b) uniform 1000-point grid vs exact search, seed 0 fold8
uni,_,_ = run_seed(0, grid="uniform", select="fold8")
exa,_,_ = run_seed(0, grid="scores",  select="fold8")
U={r["cls"]:r for r in uni}; E={r["cls"]:r for r in exa}
with open(OUT/"supp_b_grid_comparison.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["class","tier","alpha","n_pos_fold9",
                "uniform_status","uniform_lambda","uniform_test_fnr","uniform_flag_rate",
                "exact_status","exact_lambda","exact_test_fnr","exact_flag_rate","changed"])
    for c in sorted(U, key=lambda c:(ORD[U[c]["tier"]], c)):
        u,e = U[c], E[c]
        w.writerow([c, u["tier"], u["alpha"], u["n_val_pos"],
                    u["status"], fmt_lambda(u["lambda_c"]), round(u["test_fnr"],6), round(u["flag_rate"],6),
                    e["status"], fmt_lambda(e["lambda_c"]), round(e["test_fnr"],6), round(e["flag_rate"],6),
                    "yes" if u["status"]!=e["status"] else ""])
cu = sum(1 for r in uni if r["status"]=="certified"); tu = sum(1 for r in uni if r["status"]=="trivial")
ce = sum(1 for r in exa if r["status"]=="certified"); te = sum(1 for r in exa if r["status"]=="trivial")
print(f"(b) uniform grid: {cu} certified, {tu} trivial | exact search: {ce} certified, {te} trivial")
print("    status changes:", [c for c in U if U[c]["status"]!=E[c]["status"]])
