"""
src/utils/hierarchy.py
----------------------
PTB-XL diagnostic hierarchy for Contribution 2.

Maps 23 subdiagnostic classes (from the 35-class model, non-rhythm only) to
their 5-class superdiagnostic parents using the diagnostic_subclass column in
scp_statements.csv.

Column order -- two DIFFERENT orders are in play, and they must never be mixed
by position:

  * ``SUPERDIAG_ORDER`` (this module, and the noisy-OR parent columns 23-27 of
    the joint matrix) is  NORM, MI, STTC, CD, HYP.
  * The trained **5-class model** emits columns in ALPHABETICAL order,
    CD, HYP, MI, NORM, STTC, because ``aggregate_labels`` sorts them. Its true
    order is recorded in that run's ``class_order.txt``.

These are not the same permutation. Combining 5-class scores with this
hierarchy by position silently mis-pairs every group (it showed up as a parent
AUROC of 0.513 instead of 0.914). Always map by name with
:func:`super_column_index`, which asserts on a name mismatch.

Rhythm classes (AFIB, AFLT, BIGU, PACE, PSVT, SARRH, SBRAD, SR, STACH,
SVARR, SVTAC, TRIGU) have no superdiagnostic parent and are excluded from
consistency checks entirely.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Internal order for this module and for noisy-OR parent columns 23-27.
# NOT the 5-class model's column order -- that is alphabetical; see the module
# docstring and use super_column_index() to convert.
SUPERDIAG_ORDER = ["NORM", "MI", "STTC", "CD", "HYP"]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def prune_parent_only(
    child_pred: "np.ndarray",
    parent_pred: "np.ndarray",
    diagnostic_groups: list[dict],
) -> "np.ndarray":
    """Drop a parent flag when no child of that group is flagged.

    Deterministic repair of parent-implies-child violations. Returns a new
    parent prediction matrix; the input is not modified. Child predictions are
    untouched, so child-implies-parent violations are neither created nor fixed
    here (under noisy-OR they cannot occur -- see below).

    Equivalence to max-child
    ------------------------
    Under noisy-OR, ``p_parent = 1 - prod(1 - p_child) >= max_child``. So if any
    child clears ``lambda`` then the noisy-OR parent clears it too, and after
    pruning the parent is flagged **iff** some child is flagged -- which is
    exactly the max-child rule at the same threshold. Pruned noisy-OR and
    max-child therefore produce identical prediction sets, and differ only in
    the parent *score* (hence in parent AUROC and ECE, which are threshold-free).
    ``tests/test_hierarchy_pruning.py`` pins this.

    Parameters
    ----------
    child_pred  : (N, 23) bool -- subdiagnostic predictions at some threshold
    parent_pred : (N, 5)  bool -- superdiagnostic predictions, columns in
                  ``SUPERDIAG_ORDER`` (use :func:`super_column_index` to convert
                  a 5-class model's own column order)
    diagnostic_groups : from :func:`build_hierarchy`

    Returns
    -------
    (N, 5) bool -- parent predictions with parent-only flags removed.
    """
    import numpy as _np

    out = _np.array(parent_pred, dtype=bool, copy=True)
    for g in diagnostic_groups:
        has_child = _np.asarray(child_pred, dtype=bool)[:, g["child_idx_sub"]].any(axis=1)
        out[:, g["parent_idx_super"]] &= has_child
    return out


def super_column_index(model_class_order: "list[str] | str | Path") -> list[int]:
    """Index array mapping a 5-class model's columns into ``SUPERDIAG_ORDER``.

    Use as ``scores[:, super_column_index(order)]`` so that column *i* of the
    result is ``SUPERDIAG_ORDER[i]``, whatever order the model itself used.

    Parameters
    ----------
    model_class_order : list of names, or a path to a ``class_order.txt``
        written by ``src/retrain.py`` (one class name per line).

    Raises
    ------
    AssertionError
        If the supplied names are not exactly the five superdiagnostic classes.
        Mapping by position instead of by name silently mis-pairs every group,
        so this is deliberately fatal rather than a warning.
    """
    if isinstance(model_class_order, (str, Path)):
        model_class_order = Path(model_class_order).read_text().split()
    names = list(model_class_order)
    assert set(names) == set(SUPERDIAG_ORDER), (
        f"5-class column names {names} do not match the superdiagnostic set "
        f"{SUPERDIAG_ORDER}; cannot map by name."
    )
    assert len(names) == len(SUPERDIAG_ORDER), (
        f"expected {len(SUPERDIAG_ORDER)} columns, got {len(names)}"
    )
    return [names.index(c) for c in SUPERDIAG_ORDER]


def build_hierarchy(
    scp_path: str | Path | None = None,
    class_names_path: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Return and optionally print the PTB-XL diagnostic hierarchy.

    Derives the mapping exclusively from scp_statements.csv — nothing
    is hardcoded.

    Parameters
    ----------
    scp_path        : path to data/ptbxl/scp_statements.csv
    class_names_path: path to checkpoints/acuity_labels/class_names.json
    verbose         : if True, print the verified mapping before returning

    Returns
    -------
    dict with keys:
        superdiag_order   : list[str]  — 5-class names in 5-class-model order
        subdiag_names     : list[str]  — 23 non-rhythm classes (indices 0-22 of
                            the 35-class model output)
        diagnostic_groups : list[dict] — one entry per superdiag group:
            {
              "parent":      str         — superdiag class name,
              "children":    list[str]   — subdiag children present in our 23,
              "parent_idx_super": int    — column index in the 5-class model,
              "child_idx_sub":    list[int] — column indices in the 23-class
                                   subdiag block (== 35-class model cols 0-22),
            }
        child_to_parent   : dict[str, str] — child name → parent name
    """
    if scp_path is None:
        scp_path = _REPO_ROOT / "data" / "ptbxl" / "scp_statements.csv"
    if class_names_path is None:
        class_names_path = (
            _REPO_ROOT / "checkpoints" / "acuity_labels" / "class_names.json"
        )

    scp = pd.read_csv(scp_path, index_col=0)
    with open(class_names_path) as f:
        all_names: list[str] = json.load(f)

    # The 35-class model output is ordered: [23 subdiag (indices 0-22)] +
    # [12 rhythm (indices 23-34)].  Confirmed from class_names.json layout.
    subdiag_names: list[str] = all_names[:23]
    subdiag_idx   = {c: i for i, c in enumerate(subdiag_names)}

    # Build subclass → parent lookup using the diagnostic_subclass column.
    # Multiple raw SCP codes (e.g. ISCAL, ISCAS, ISCLA, ISCAN) aggregate to
    # one label (ISCA), which maps to a superdiagnostic parent (STTC).
    subclass_to_parent: dict[str, str] = {}
    for _, row in scp.iterrows():
        dc = row.get("diagnostic_class", "")
        ds = row.get("diagnostic_subclass", "")
        if pd.notna(dc) and pd.notna(ds) and str(dc) and str(ds):
            subclass_to_parent[str(ds)] = str(dc)

    # Verify exactly 5 non-null superdiagnostic groups
    unique_parents = set(subclass_to_parent.values())
    assert unique_parents == set(SUPERDIAG_ORDER), (
        f"Expected superdiagnostic groups {SUPERDIAG_ORDER}, "
        f"got {sorted(unique_parents)}"
    )

    # Build groups restricted to our 23 subdiag classes
    superdiag_idx_5 = {p: i for i, p in enumerate(SUPERDIAG_ORDER)}
    child_to_parent: dict[str, str] = {}
    superdiag_to_children: dict[str, list[str]] = {p: [] for p in SUPERDIAG_ORDER}

    for cls in subdiag_names:
        parent = subclass_to_parent.get(cls)
        if parent is None:
            raise ValueError(
                f"Subdiagnostic class '{cls}' has no parent in scp_statements.csv. "
                "Check that it is not a rhythm class placed in the wrong block."
            )
        superdiag_to_children[parent].append(cls)
        child_to_parent[cls] = parent

    diagnostic_groups = [
        {
            "parent":           parent,
            "children":         superdiag_to_children[parent],
            "parent_idx_super": superdiag_idx_5[parent],
            "child_idx_sub":    [subdiag_idx[c] for c in superdiag_to_children[parent]],
        }
        for parent in SUPERDIAG_ORDER
    ]

    if verbose:
        print("=" * 60)
        print("PTB-XL hierarchy (derived from scp_statements.csv)")
        print("=" * 60)
        print(f"{'Group':<6}  {'5-cl idx':>8}  Children (23-cl model indices)")
        print("-" * 60)
        for grp in diagnostic_groups:
            child_str = ", ".join(
                f"{c}({grp['child_idx_sub'][i]})"
                for i, c in enumerate(grp["children"])
            )
            print(
                f"{grp['parent']:<6}  "
                f"{grp['parent_idx_super']:>8}  "
                f"{child_str}"
            )
        print("=" * 60)
        print(f"Groups: {len(diagnostic_groups)}  |  "
              f"Total subdiag covered: "
              f"{sum(len(g['children']) for g in diagnostic_groups)}/23")
        print()

    return {
        "superdiag_order":   SUPERDIAG_ORDER,
        "subdiag_names":     subdiag_names,
        "diagnostic_groups": diagnostic_groups,
        "child_to_parent":   child_to_parent,
    }
