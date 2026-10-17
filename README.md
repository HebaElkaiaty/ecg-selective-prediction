# Acuity-Aware Selective Prediction for Multi-Label ECG Classification

Post-hoc calibration and selective prediction on PTB-XL. Class-conditional
false-negative-rate control with acuity-dependent targets, certified by
Learn-then-Test, plus a comparison of hierarchy-consistency methods.

Code for the paper. Results are not version-controlled — they regenerate from
saved model outputs in about 70 seconds.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11. Training uses Apple Silicon `mps` where available, otherwise CPU.

## Data

PTB-XL **v1.0.3** (21,799 records) from PhysioNet, unpacked to `data/ptbxl/`:

```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

v1.0.3 removed 38 duplicate records relative to v1.0.1, so counts differ
slightly from papers using the earlier release.

## Run

```bash
# 1. train (per model, per seed; ~3 min each on an M3 Max)
python -m src.retrain --model acuity35 --seed 0 --select-on-fold8
python -m src.retrain --model super5   --seed 0 --select-on-fold8

# 2. every table and figure, from the saved outputs
python -m src.revision.run_all

# tests
python -m pytest tests/ -q
```

`run_all` writes CSVs to `results/revision/` and figures to `results/figures/`.
It also carries an optional regression gate that diffs the calibration driver
against a previously published threshold table; that table is not distributed
here, so the step reports itself skipped.

Paths are overridable: `ECG_SEED`, `ECG_SELECT`, `ECG_CKPT`, `ECG_OUT`, `ECG_FIG`.

## Reproducibility

| | |
|---|---|
| **Splits** | folds 1–7 train, fold 8 checkpoint selection, fold 9 calibration, fold 10 test (15,245 / 2,173 / 2,183 / 2,198). Folds 9 and 10 are not loaded during training. |
| **Seeds** | 0–4; seed 0 fixed a priori as the primary model. Each run records its configuration in `run_config.json` and its per-epoch history in `history.csv`. |
| **Checkpoint selection** | epoch with the highest macro AUROC on fold 8. No fold-9 or fold-10 quantity influences selection. |
| **Label mapping** | derived from `scp_statements.csv` by `src/utils/labels_acuity.py`. The 35 model classes with their column order, acuity tier and target α are in `src/calibration/acuity_tiers.csv`. |
| **Preprocessing** | one global `StandardScaler`, `mean_ = -0.0007771724002828525`, `scale_ = 0.23895642180464072`, so the transform is `(x - mean)/scale`. `src/utils/data.py::load_standardizer` uses the benchmark pickle when present and otherwise rebuilds these parameters; the two are bit-identical. Fit on the benchmark's training folds at 500 Hz, applied here at 100 Hz; folds 9 and 10 never informed it. |
| **Thresholds** | candidates are the observed fold-9 positive scores (exact search). Because FNR is monotone in λ only the boundary is tested, so the candidate set carries no multiplicity cost. A uniform 1,000-point grid is retained for one supplementary comparison. |
| **Multiplicity** | Bonferroni across the 33 non-excluded classes, fixed a priori: δ_k = 0.10/33. |
| **Calibrated thresholds** | `calibrated_thresholds.csv` ships with this repository: all 33 classes across seeds 0–4, with status, λ, test misses, FNR and flag rate. Regenerate with `python -m src.revision.export_thresholds`. |

## Layout

```
config.py                     paths and hyperparameters
src/retrain.py                training, both models, parameterised by seed
src/calibrate_acuity.py       LTT calibration driver
src/calibration/              risk control, hierarchy consistency, tier table
src/models/resnet1d.py        backbone
src/utils/                    data loading, label generation, hierarchy
src/revision/                 analyses, figures, run_all
calibrated_thresholds.csv     all thresholds, 33 classes x 5 seeds
tests/                        synthetic-data unit tests
```

## License

Code under MIT. PTB-XL is distributed by PhysioNet under its own license.
