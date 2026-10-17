from pathlib import Path

_ROOT = Path(__file__).parent

PTBXL_PATH     = _ROOT / "data" / "ptbxl"
# Optional. If the PTB-XL benchmarking repository's standard_scaler.pkl is
# present here it is used; otherwise src/utils/data.load_standardizer rebuilds
# an identical scaler from its published parameters.
HELME_DATA     = _ROOT / "output" / "ecg_ptbxl_benchmarking" / "exp0" / "data"
CHECKPOINT_DIR = str(_ROOT / "checkpoints")
SAMPLING_RATE  = 100
BATCH_SIZE     = 64
EPOCHS         = 30
LR             = 1e-3
# Superdiagnostic classes. The trained model's COLUMN ORDER is alphabetical --
# CD, HYP, MI, NORM, STTC -- because aggregate_labels() sorts them; each run
# records it in class_order.txt. This is NOT the order used by
# src/utils/hierarchy.SUPERDIAG_ORDER (NORM, MI, STTC, CD, HYP); convert with
# hierarchy.super_column_index() rather than by position.
NUM_CLASSES    = 5
