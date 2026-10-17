"""Run and output locations for the revision analyses.

Results are deliberately NOT version-controlled: they change with every run, and
keeping them out of the tree lets the code stay general. Nothing here may be
hardcoded to one run -- every path is overridable, so the same scripts work for
any seed or checkpoint rule.

Override with environment variables:

    ECG_SEED=3 python -m src.revision.run_all
    ECG_SELECT=final ECG_OUT=results/revision_final python -m src.revision.run_all
    ECG_CKPT=/mnt/scratch/retrain python -m src.revision.e2_per_class

| Variable     | Default               | Meaning                                  |
|--------------|-----------------------|------------------------------------------|
| ECG_CKPT     | checkpoints/retrain   | directory holding the per-run folders     |
| ECG_SEED     | 0                     | primary seed (fixed a priori; see Methods)|
| ECG_SELECT   | fold8                 | checkpoint rule: fold8 | final            |
| ECG_OUT      | results/revision      | CSV output directory                      |
| ECG_FIG      | results/figures       | figure output directory                   |
"""
from __future__ import annotations

import os
from pathlib import Path

CKPT_ROOT = Path(os.environ.get("ECG_CKPT", "checkpoints/retrain"))
SEED      = int(os.environ.get("ECG_SEED", "0"))
SELECT    = os.environ.get("ECG_SELECT", "fold8")
OUT       = Path(os.environ.get("ECG_OUT", "results/revision"))
FIG       = Path(os.environ.get("ECG_FIG", "results/figures"))

# Seeds used for the training-variability tables. Folds 9 and 10 are shared
# across seeds, so this spread measures training variability, not the guarantee.
SEEDS = [int(s) for s in os.environ.get("ECG_SEEDS", "0,1,2,3,4").split(",")]


def run_dir(model: str = "acuity35", seed: int | None = None,
            select: str | None = None) -> Path:
    """Directory for one run, e.g. checkpoints/retrain/acuity35_seed0_fold8."""
    return CKPT_ROOT / f"{model}_seed{SEED if seed is None else seed}_{SELECT if select is None else select}"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


ensure_dirs()


def fmt_lambda(x) -> str:
    """Serialise a threshold without destroying it.

    Certified thresholds span ten orders of magnitude (the smallest observed is
    1.5e-10). round(x, 6) collapses anything below 5e-7 to 0.0, which is then
    indistinguishable from "no threshold" -- a consumer would flag every record
    instead of almost none. 12 significant digits round-trips float64 safely.
    """
    return f"{float(x):.12g}"
