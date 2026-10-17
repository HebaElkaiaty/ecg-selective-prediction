"""
src/utils/labels_acuity.py
--------------------------
Contribution 1: generate the combined 35-class (23 subdiagnostic + 12 rhythm)
multi-hot label matrix used for acuity-weighted class-conditional risk control.

Column layout
-------------
Columns 0–22  : subdiagnostic classes (aggregate_labels task='sub'), alphabetical
Columns 23–34 : rhythm classes        (aggregate_labels task='rhythm'), alphabetical

An all-zero row is a valid "no flagged sub/rhythm finding" — do not exclude it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure repo root is importable whether this file is run as a script or imported
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from src.utils.data import aggregate_labels


def generate_combined_labels(
    meta: pd.DataFrame,
    scp_statements: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """Build combined subdiagnostic + rhythm multi-hot matrix.

    Parameters
    ----------
    meta :
        DataFrame from load_ptbxl_metadata (must contain scp_codes column).
    scp_statements :
        scp_statements.csv loaded with index_col=0.

    Returns
    -------
    matrix : np.ndarray, shape (N, 35), dtype float32
    class_names : list of str, length 35
        class_names[i] is the label for column i of matrix.
        Subdiagnostic names occupy indices 0–22; rhythm names 23–34.
    """
    meta_cols = set(meta.columns) | {"label_primary"}

    sub_df = aggregate_labels(meta, scp_statements, task="sub")
    sub_names = sorted(c for c in sub_df.columns if c not in meta_cols)
    sub_matrix = sub_df[sub_names].values.astype(np.float32)

    rhy_df = aggregate_labels(meta, scp_statements, task="rhythm")
    rhy_names = sorted(c for c in rhy_df.columns if c not in meta_cols)
    rhy_matrix = rhy_df[rhy_names].values.astype(np.float32)

    combined = np.concatenate([sub_matrix, rhy_matrix], axis=1)
    class_names = sub_names + rhy_names

    return combined, class_names


# ---------------------------------------------------------------------------
# CLI: save outputs and print per-class counts
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import CHECKPOINT_DIR, PTBXL_PATH
    from src.utils.data import get_fold_splits, load_ptbxl_metadata

    meta = load_ptbxl_metadata(PTBXL_PATH)
    scp = pd.read_csv(Path(PTBXL_PATH) / "scp_statements.csv", index_col=0)
    splits = get_fold_splits(meta)

    Y, class_names = generate_combined_labels(meta, scp)

    assert Y.shape == (len(meta), 35), f"Expected (N,35), got {Y.shape}"
    assert len(class_names) == 35

    out_dir = Path(CHECKPOINT_DIR) / "acuity_labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "y_combined_train.npy", Y[splits["train"]])
    np.save(out_dir / "y_combined_val.npy",   Y[splits["val"]])
    np.save(out_dir / "y_combined_test.npy",  Y[splits["test"]])
    with open(out_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"Saved to {out_dir}/")
    print(f"Total classes: {len(class_names)}")
    print()

    y_tr = Y[splits["train"]]
    y_va = Y[splits["val"]]
    y_te = Y[splits["test"]]
    n_tr, n_va, n_te = len(splits["train"]), len(splits["val"]), len(splits["test"])

    header = f"{'#':<4}{'Class':<16}  {'Train+':>8}/{n_tr:<6}  {'Val+':>6}/{n_va:<5}  {'Test+':>6}/{n_te}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        t_pos  = int(y_tr[:, i].sum())
        va_pos = int(y_va[:, i].sum())
        te_pos = int(y_te[:, i].sum())
        tag = "  [sub]" if i < 23 else "  [rhy]"
        print(f"{i:<4}{name:<16}  {t_pos:>8}/{n_tr:<6}  {va_pos:>6}/{n_va:<5}  {te_pos:>6}/{n_te}{tag}")

    # Sanity: a 10-second strip is either in normal sinus rhythm or a competing
    # primary rhythm — these should rarely co-occur in the same record.
    # "Confirmed abnormal" = AFIB, AFLT, SVTAC, STACH at likelihood == 100.
    confirmed_abnormal = {"AFIB", "AFLT", "SVTAC", "STACH"}
    sr_idx = class_names.index("SR")
    scp_list = meta["scp_codes"].tolist()
    co_occur = sum(
        1
        for i, scps in enumerate(scp_list)
        if Y[i, sr_idx] == 1.0
        and any(scps.get(c, 0.0) == 100.0 for c in confirmed_abnormal)
    )
    sr_total = int((Y[:, sr_idx] == 1.0).sum())
    print()
    print("Sanity — SR-positive records that also carry a confirmed abnormal rhythm (likelihood=100):")
    print(f"  AFIB/AFLT/SVTAC/STACH @100 co-occurring with SR: {co_occur} / {sr_total} SR-positive records")
