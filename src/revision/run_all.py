"""Regenerate every revision table and figure, in dependency order.

Reads only saved model outputs under ECG_CKPT; writes only to ECG_OUT / ECG_FIG.
Results are not version-controlled -- this script is how they are reproduced.

    python -m src.revision.run_all              # everything (seed 0, fold8)
    python -m src.revision.run_all --skip-sim   # seconds; omits the simulation
    ECG_SEED=3 python -m src.revision.run_all   # any other seed

Not included: dump_fold8 (needs the 1 GB raw signal cache) and retrain.py
(training is launched by hand). Both are run explicitly.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
import time

from src.revision.paths import CKPT_ROOT, FIG, OUT, SEED, SELECT, run_dir

# (module, what it produces) -- order matters: later steps read earlier CSVs.
STEPS = [
    ("verify_legacy_repro",     "regression gate: reproduce the SUBMITTED results"),
    ("e2_per_class",            "E2 per-class table, seed variability, per-seed AUROC"),
    ("e2b_class_auroc",         "per-class AUROC on fold 10, all 33 tiered classes"),
    ("supp_tables",             "S1 final-epoch, S2 uniform grid vs exact search"),
    ("e_baselines",             "Tables 3/4 uncorrected baselines, temperature and ECE"),
    ("e_sensitivity",           "Table 2 tier-alpha sensitivity, Table 6 uniform vs tiered"),
    ("e_crc",                   "E3 conformal risk control"),
    ("e5_e7_variants",          "E5 tier reassignment, E7 delta = 0.05"),
    ("e_contrib2",              "Table 8 hierarchy sweep, Figure 4"),
    ("e6b_contrib2_detail",     "C2 grid detail and certified-interval structure"),
    ("e6c_hierarchy_baselines", "E6 hierarchy baselines"),
    ("e6d_table9",              "Table 9 consistency comparison with CP intervals"),
    ("e11_crc_configurations",  "Table S5 CRC at tiered and uniform targets"),
    ("e8_aflt",                 "E8 AFLT support by fold"),
    ("e9_multiplicity",         "E9 Holm/BH corrections, sample-size requirement"),
    ("e4_validity_sim",         "E4 validity simulation (2,000 bootstrap trials)"),
    ("e10_estimator_and_dependence",
                                "E10 estimator variants, selection-dependence arm"),
    ("e_figures",               "Figures 1-3"),
    ("make_supplementary_tex",  "Supplementary Tables S1-S5 (LaTeX)"),
]

# Document-generation steps, present only in the authoring checkout.
WRITING_STEPS = [
    ("make_handoff",            "writing brief"),
    ("make_insert_values",      "manuscript values (first batch)"),
    ("append_insert_values",    "manuscript values (second batch)"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-sim", action="store_true",
                    help="omit the two bootstrap simulations (minutes rather than "
                         "seconds)")
    ap.add_argument("--only", nargs="*", help="run only these module names")
    args = ap.parse_args()

    missing = [p for p in (run_dir("acuity35"), run_dir("super5")) if not p.exists()]
    if missing:
        print("Missing run directories -- train first with src/retrain.py:")
        for p in missing:
            print(f"  {p}")
        return 1

    here = Path(__file__).parent
    SIMS = {"e4_validity_sim", "e10_estimator_and_dependence"}
    steps = [s for s in STEPS if not (args.skip_sim and s[0] in SIMS)]
    steps += [s for s in WRITING_STEPS if (here / f"{s[0]}.py").exists()]
    if args.only:
        steps = [s for s in steps if s[0] in args.only]

    print(f"ckpt={CKPT_ROOT}  seed={SEED}  select={SELECT}  out={OUT}  fig={FIG}")
    t0, failed = time.time(), []
    for i, (mod, what) in enumerate(steps, 1):
        t = time.time()
        print(f"[{i}/{len(steps)}] {mod} -- {what}", flush=True)
        try:
            m = importlib.import_module(f"src.revision.{mod}")
            if hasattr(m, "main"):
                m.main()
        except Exception as exc:                       # noqa: BLE001
            failed.append((mod, exc)); print(f"    FAILED: {exc}")
        else:
            print(f"    done in {time.time()-t:.1f}s")
    print(f"\n{len(steps)-len(failed)}/{len(steps)} steps in {time.time()-t0:.1f}s")
    for mod, exc in failed:
        print(f"  FAILED {mod}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
