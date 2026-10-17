"""
src/utils/data.py
-----------
PTB-XL dataset loading, label aggregation, signal loading,
fold splitting, and signal standardisation.

References
----------
Wagner et al. (2020) PTB-XL, a large publicly available
electrocardiography dataset. Scientific Data 7, 154.

Strodthoff et al. (2021) Deep Learning for ECG Analysis:
Benchmarks and Insights from PTB-XL. IEEE Journal of
Biomedical and Health Informatics 25(5), 1519-1528.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List

import pickle

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRAIN_FOLDS: List[int] = list(range(1, 9))   # folds 1–8
_VAL_FOLDS:   List[int] = [9]
_TEST_FOLDS:  List[int] = [10]

# PTB-XL encodes confirmed findings at likelihood=0.0 for rhythm codes and
# for NST_ (non-specific ST changes). For consistency, aggregate_labels uses
# presence-in-dict as the universal positivity signal: any code that appears
# in scp_codes is positive regardless of its stored likelihood value.

# Columns kept from ptbxl_database.csv after cleaning
_META_COLS = [
    "ecg_id",
    "patient_id",
    "age",
    "sex",
    "strat_fold",
    "scp_codes",
    "filename_hr",   # 500 Hz WFDB path
    "filename_lr",   # 100 Hz WFDB path
]

# Mapping from task name → column in scp_statements.csv that provides the
# class label for each SCP code.  NaN means the code is irrelevant for that
# task and should be ignored.
_TASK_COL: Dict[str, str] = {
    "super":   "diagnostic_class",
    "sub":     "diagnostic_subclass",
    "rhythm":  "rhythm",
    "form":    "form",
}


# ---------------------------------------------------------------------------
# 1. Metadata loading
# ---------------------------------------------------------------------------

def load_ptbxl_metadata(data_dir: Path) -> pd.DataFrame:
    """Load and clean ``ptbxl_database.csv``.

    Parameters
    ----------
    data_dir:
        Root directory of the PTB-XL dataset (the folder that contains
        ``ptbxl_database.csv``, ``scp_statements.csv``, and the ``records*``
        sub-directories).

    Returns
    -------
    pd.DataFrame
        One row per recording with columns:
        ``ecg_id``, ``patient_id``, ``age``, ``sex``, ``strat_fold``,
        ``scp_codes`` (Python *dict*), ``filename_hr``, ``filename_lr``.
    """
    csv_path = Path(data_dir) / "ptbxl_database.csv"
    df = pd.read_csv(csv_path, index_col="ecg_id")
    df.index.name = "ecg_id"
    df = df.reset_index()

    # scp_codes is stored as a string-encoded Python dict, e.g.
    # "{'NORM': 100.0, 'SR': 0.0}" — parse it safely with ast.literal_eval.
    df["scp_codes"] = df["scp_codes"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else {}
    )

    # Keep only the columns we care about
    available = [c for c in _META_COLS if c in df.columns]
    df = df[available].copy()

    # Coerce numeric types
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["strat_fold"] = df["strat_fold"].astype(int)

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Label aggregation
# ---------------------------------------------------------------------------

def aggregate_labels(
    metadata: pd.DataFrame,
    scp_statements: pd.DataFrame,
    task: str,
) -> pd.DataFrame:
    """Map SCP codes to task-specific class labels.

    For each recording the ``scp_codes`` dict may contain several SCP codes
    with associated likelihood values (0–100).  This function:

    1. Looks up the relevant label column in ``scp_statements`` for the
       chosen *task*.
    2. Keeps only codes that have a non-null label for that task.
    3. Returns, for every recording, the **set of applicable labels**
       (multi-label, binary indicator columns) as well as the single label
       with the highest likelihood (``label_primary``).

    Parameters
    ----------
    metadata:
        DataFrame returned by :func:`load_ptbxl_metadata`.
    scp_statements:
        ``scp_statements.csv`` loaded with ``index_col=0``
        (SCP code strings as the index).
    task:
        One of ``'super'``, ``'sub'``, ``'rhythm'``, ``'form'``.

    Returns
    -------
    pd.DataFrame
        Same index as *metadata* plus:

        * One binary column per unique class label (multi-hot encoding).
        * ``label_primary``: the highest-likelihood label (str) or ``NaN``
          if no label applies.
    """
    if task not in _TASK_COL:
        raise ValueError(
            f"Unknown task '{task}'. Choose from {list(_TASK_COL)}."
        )

    label_col = _TASK_COL[task]

    # Build SCP → class mapping for this task (drop NaN entries)
    if label_col not in scp_statements.columns:
        raise KeyError(
            f"Column '{label_col}' not found in scp_statements. "
            f"Available columns: {scp_statements.columns.tolist()}"
        )

    scp_to_label: Dict[str, str] = (
        scp_statements[label_col]
        .dropna()
        .astype(str)
        .to_dict()
    )

    # When the column is a numeric-presence flag (e.g. rhythm=1.0), every
    # code maps to the same value — use the SCP code itself as the class name.
    if len(set(scp_to_label.values())) == 1:
        scp_to_label = {code: code for code in scp_to_label}

    # For each recording determine which labels apply (any likelihood > 0)
    # and which label has the highest likelihood.
    all_labels = sorted(set(scp_to_label.values()))

    rows = []
    for _, row in metadata.iterrows():
        scp_codes: dict = row["scp_codes"]

        # Collect (label, likelihood) for codes that map to a class
        label_likelihoods: Dict[str, float] = {}
        for code, likelihood in scp_codes.items():
            if code in scp_to_label:
                lbl = scp_to_label[code]
                # If multiple SCP codes map to the same label take the max
                label_likelihoods[lbl] = max(
                    label_likelihoods.get(lbl, 0.0), float(likelihood)
                )

        # Multi-hot vector
        indicator = {lbl: int(lbl in label_likelihoods) for lbl in all_labels}

        # Primary label: highest likelihood; NaN when nothing applies
        if label_likelihoods:
            indicator["label_primary"] = max(
                label_likelihoods, key=label_likelihoods.__getitem__
            )
        else:
            indicator["label_primary"] = np.nan

        rows.append(indicator)

    label_df = pd.DataFrame(rows, index=metadata.index)
    return pd.concat([metadata, label_df], axis=1)


# ---------------------------------------------------------------------------
# 3. Signal loading
# ---------------------------------------------------------------------------

def load_signals(
    metadata: pd.DataFrame,
    data_dir: Path,
    sampling_rate: int = 100,
) -> np.ndarray:
    """Load raw ECG waveforms from WFDB records.

    Parameters
    ----------
    metadata:
        DataFrame returned by :func:`load_ptbxl_metadata`.
    data_dir:
        Root directory of the PTB-XL dataset.
    sampling_rate:
        Either ``100`` (uses ``filename_lr``) or ``500`` (uses
        ``filename_hr``).  Defaults to ``100``.

    Returns
    -------
    np.ndarray
        Shape ``(n_records, n_samples, 12)``, dtype ``float32``.
        ``n_samples`` is ``1000`` at 100 Hz or ``5000`` at 500 Hz.

    Raises
    ------
    ValueError
        If *sampling_rate* is not 100 or 500.
    FileNotFoundError
        If a WFDB header file cannot be found.
    """
    data_dir = Path(data_dir)

    if sampling_rate == 100:
        filename_col = "filename_lr"
    elif sampling_rate == 500:
        filename_col = "filename_hr"
    else:
        raise ValueError("sampling_rate must be 100 or 500.")

    if filename_col not in metadata.columns:
        raise KeyError(
            f"Column '{filename_col}' not found in metadata. "
            "Did you call load_ptbxl_metadata correctly?"
        )

    signals = []
    for _, row in metadata.iterrows():
        record_path = data_dir / row[filename_col]
        signal, _fields = wfdb.rdsamp(str(record_path))
        # signal shape: (n_samples, 12)
        signals.append(signal.astype(np.float32))

    # Stack to (n_records, n_samples, 12)
    arr = np.stack(signals, axis=0)
    return arr


# ---------------------------------------------------------------------------
# 4. Fold splits
# ---------------------------------------------------------------------------

def get_fold_splits(metadata: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Return integer index arrays for train / val / test splits.

    Uses the Strodthoff et al. convention:

    * **Train** : folds 1–8
    * **Val**   : fold 9
    * **Test**  : fold 10

    Parameters
    ----------
    metadata:
        DataFrame returned by :func:`load_ptbxl_metadata`.  Must contain a
        ``strat_fold`` column with integer values 1–10.

    Returns
    -------
    dict with keys ``'train'``, ``'val'``, ``'test'``; values are
    1-D ``np.ndarray`` of integer *positional* indices (suitable for
    ``arr[splits['train']]``).
    """
    if "strat_fold" not in metadata.columns:
        raise KeyError("'strat_fold' column not found in metadata.")

    folds = metadata["strat_fold"].values

    return {
        "train": np.where(np.isin(folds, _TRAIN_FOLDS))[0],
        "val":   np.where(np.isin(folds, _VAL_FOLDS))[0],
        "test":  np.where(np.isin(folds, _TEST_FOLDS))[0],
    }


# ---------------------------------------------------------------------------
# 5. Standardisation
# ---------------------------------------------------------------------------

def fit_standardizer(X_train: np.ndarray) -> StandardScaler:
    """Fit a per-channel zero-mean, unit-variance scaler on the training set.

    Parameters
    ----------
    X_train:
        Array of shape ``(n_train, n_samples, 12)``.

    Returns
    -------
    sklearn.preprocessing.StandardScaler
        Fitted on a ``(n_train * n_samples, 12)`` view of the data so that
        mean and variance are computed per lead across all time steps and
        training recordings.
    """
    n, t, c = X_train.shape
    # Reshape to (n*t, 12) so sklearn sees one sample per time step
    X_flat = X_train.reshape(-1, c)
    scaler = StandardScaler()
    scaler.fit(X_flat)
    return scaler


# Parameters of the single global StandardScaler distributed with the PTB-XL
# benchmarking repository (Strodthoff et al.), published here so the
# preprocessing is reproducible without that artifact. It is a one-feature
# scaler, so the transform is exactly (x - mean) / scale.
#
# n_samples_seen_ = 1_046_460_000 = 17_441 x 5_000 x 12, i.e. it was fit on the
# benchmark's training folds (1-8) at 500 Hz and is applied here at 100 Hz.
# Folds 9 and 10 never informed it.
HELME_SCALER_MEAN  = -0.0007771724002828525
HELME_SCALER_SCALE = 0.23895642180464072


def load_standardizer(path: "str | Path | None" = None) -> StandardScaler:
    """Load the benchmark scaler, or rebuild it from the published constants.

    Uses the pickle at *path* when it exists; otherwise returns a scaler with
    identical parameters. The two give bit-identical output, so a replicator
    without the benchmark repository reproduces the preprocessing exactly.
    """
    if path is not None and Path(path).exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    s = StandardScaler()
    s.mean_ = np.array([HELME_SCALER_MEAN])
    s.scale_ = np.array([HELME_SCALER_SCALE])
    s.var_ = s.scale_ ** 2
    s.n_features_in_ = 1
    s.n_samples_seen_ = 1_046_460_000
    s.with_mean = s.with_std = True
    return s


def apply_standardizer(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply a fitted scaler to ECG data.

    Parameters
    ----------
    X:
        Array of shape ``(n, n_samples, 12)``.
    scaler:
        A :class:`~sklearn.preprocessing.StandardScaler` previously returned
        by :func:`fit_standardizer`.

    Returns
    -------
    np.ndarray
        Standardised array of the same shape as *X*, dtype ``float32``.
    """
    n, t, c = X.shape
    n_features = scaler.mean_.shape[0]
    if n_features == c:
        X_flat = X.reshape(-1, c)
    elif n_features == 1:
        # helme's scaler was fit globally (single mean/std across all channels)
        X_flat = X.reshape(-1, 1)
    else:
        raise ValueError(
            f"Scaler has {n_features} features; expected 1 (global) or {c} (per-channel)."
        )
    X_scaled = scaler.transform(X_flat).reshape(n, t, c)
    return X_scaled.astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience: load everything in one call
# ---------------------------------------------------------------------------

def load_ptbxl(
    data_dir: str | Path,
    task: str = "super",
    sampling_rate: int = 100,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, np.ndarray], StandardScaler]:
    """High-level helper that wires together all steps.

    Returns
    -------
    X : np.ndarray, shape (N, T, 12)
        Standardised signals (train-set statistics).
    Y : np.ndarray, shape (N, n_classes)
        Multi-hot label matrix (columns ordered alphabetically).
    labelled_meta : pd.DataFrame
        Metadata + label columns.
    splits : dict
        ``{'train': idx, 'val': idx, 'test': idx}``
    scaler : StandardScaler
        Fitted standardiser (keep for inference-time use).
    """
    data_dir = Path(data_dir)

    meta = load_ptbxl_metadata(data_dir)
    scp = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)

    labelled = aggregate_labels(meta, scp, task=task)
    splits = get_fold_splits(labelled)

    X = load_signals(meta, data_dir, sampling_rate=sampling_rate)

    scaler = fit_standardizer(X[splits["train"]])
    X = apply_standardizer(X, scaler)

    # Multi-hot matrix: exclude non-label columns
    label_cols = [
        c for c in labelled.columns
        if c not in list(meta.columns) + ["label_primary"]
    ]
    Y = labelled[label_cols].values.astype(np.float32)

    return X, Y, labelled, splits, scaler
