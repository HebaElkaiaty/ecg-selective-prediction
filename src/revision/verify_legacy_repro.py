"""Optional regression gate against a previously published threshold table.

Runs the calibration driver in a legacy configuration -- a flat n+ >= 10 class
family (delta_k = 0.10/24) and a uniform 1000-point grid -- and diffs lambda_c,
test FNR, flag rate and status against a reference CSV, failing if any differ.

The reference table and the model outputs it was computed from are not
distributed with this repository, so this step is skipped unless they are
present. Where they are, it runs first: a drifted driver invalidates everything
downstream, so it is worth catching before anything else runs.
"""
from __future__ import annotations
import csv, subprocess, sys
from pathlib import Path

from src.revision.paths import OUT

REF = Path("results/contribution1/primary_calibration.csv")


def main():
    if not REF.exists():
        print(f"  skipped: no reference table at {REF}"); return
    out = OUT / "_repro_legacy24.csv"
    r = subprocess.run(
        [sys.executable, "-m", "src.calibrate_acuity", "--source", "legacy",
         "--family", "legacy24", "--grid", "uniform",
         "--validate-against", str(REF), "--out", str(out)],
        capture_output=True, text=True)
    print(r.stdout.rstrip())
    if r.returncode != 0:
        raise RuntimeError(f"driver failed:\n{r.stderr}")
    if "EXACT MATCH" not in r.stdout:
        raise AssertionError(
            "calibration driver no longer reproduces the submitted results -- "
            "downstream numbers are not trustworthy until this is resolved")
    print("  regression gate PASSED: submitted results reproduced exactly")


if __name__ == "__main__":
    main()
