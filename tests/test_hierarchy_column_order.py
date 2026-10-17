"""Regression test for the 5-class / hierarchy column-order mismatch.

The 5-class model emits alphabetical columns (CD, HYP, MI, NORM, STTC) while
hierarchy.SUPERDIAG_ORDER is NORM, MI, STTC, CD, HYP. Combining them by
position silently mis-pairs every diagnostic group and was first seen as a
parent AUROC of 0.513 instead of 0.914.
"""
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.utils.hierarchy import SUPERDIAG_ORDER, super_column_index

S5 = Path("checkpoints/retrain/super5_seed0_fold8")
needs_run = pytest.mark.skipif(
    not (S5 / "scores_fold10.npy").exists(),
    reason="seed-0 fold8 super5 run not present",
)


def test_orders_are_different_permutations():
    """Guard the premise: the two orders genuinely differ."""
    alphabetical = sorted(SUPERDIAG_ORDER)
    assert alphabetical == ["CD", "HYP", "MI", "NORM", "STTC"]
    assert alphabetical != SUPERDIAG_ORDER
    assert super_column_index(alphabetical) != list(range(5))


def test_super_column_index_maps_by_name():
    idx = super_column_index(["CD", "HYP", "MI", "NORM", "STTC"])
    assert [["CD", "HYP", "MI", "NORM", "STTC"][i] for i in idx] == SUPERDIAG_ORDER


def test_super_column_index_is_an_involution_on_round_trip():
    order = ["CD", "HYP", "MI", "NORM", "STTC"]
    idx = super_column_index(order)
    arr = np.arange(5 * 3).reshape(3, 5)
    back = np.empty_like(arr)
    back[:, idx] = arr[:, idx]
    assert np.array_equal(back[:, idx], arr[:, idx])


def test_rejects_mismatched_names():
    with pytest.raises(AssertionError):
        super_column_index(["CD", "HYP", "MI", "NORM", "NOPE"])
    with pytest.raises(AssertionError):
        super_column_index(["CD", "HYP", "MI", "NORM"])


@needs_run
def test_reproduces_e6_super5_parent_auroc():
    """Name-based mapping reproduces the reported parent AUROC of 0.9161."""
    order = (S5 / "class_order.txt").read_text().split()
    idx = super_column_index(order)
    s = np.load(S5 / "scores_fold10.npy")[:, idx]
    y = np.load(S5 / "labels_fold10.npy")[:, idx]
    auroc = float(np.mean([
        roc_auc_score(y[:, k], s[:, k])
        for k in range(5) if len(np.unique(y[:, k])) >= 2
    ]))
    assert auroc == pytest.approx(0.9161, abs=5e-4)


@needs_run
def test_positional_mixing_is_detectably_wrong():
    """The bug this guards against: positional use collapses parent AUROC."""
    order = (S5 / "class_order.txt").read_text().split()
    idx = super_column_index(order)
    s_raw = np.load(S5 / "scores_fold10.npy")          # model order
    y_mapped = np.load(S5 / "labels_fold10.npy")[:, idx]  # SUPERDIAG_ORDER
    wrong = float(np.mean([
        roc_auc_score(y_mapped[:, k], s_raw[:, k])
        for k in range(5) if len(np.unique(y_mapped[:, k])) >= 2
    ]))
    assert wrong < 0.60, f"expected near-chance when mixed by position, got {wrong}"


@needs_run
def test_macro_auroc_is_order_invariant_when_both_sides_match():
    """Item 4: the reported super5 macro-AUROC used matched orders, so it stands."""
    s = np.load(S5 / "scores_fold10.npy")
    y = np.load(S5 / "labels_fold10.npy")
    macro = lambda a, b: float(np.mean([
        roc_auc_score(b[:, k], a[:, k])
        for k in range(b.shape[1]) if len(np.unique(b[:, k])) >= 2]))
    base = macro(s, y)
    for perm in ([3, 2, 4, 0, 1], [4, 3, 2, 1, 0], [1, 0, 3, 2, 4]):
        assert macro(s[:, perm], y[:, perm]) == pytest.approx(base, abs=1e-12)
