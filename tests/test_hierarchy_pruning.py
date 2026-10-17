"""Pruned noisy-OR sets must equal max-child sets at the same threshold.

Under noisy-OR, p_parent = 1 - prod(1 - p_child) >= max_child, so any child
clearing lambda forces the parent to clear it. Pruning parent-only flags then
leaves "parent flagged iff some child flagged" -- the max-child rule exactly.
"""
from pathlib import Path

import numpy as np
import pytest

from src.utils.hierarchy import build_hierarchy, prune_parent_only, super_column_index

A35 = Path("checkpoints/retrain/acuity35_seed0_fold8")
needs_run = pytest.mark.skipif(
    not (A35 / "scores_fold10.npy").exists(), reason="seed-0 fold8 run not present"
)


def _synthetic_groups():
    return [
        {"parent_idx_super": 0, "child_idx_sub": [0, 1]},
        {"parent_idx_super": 1, "child_idx_sub": [2]},
        {"parent_idx_super": 2, "child_idx_sub": [3, 4, 5]},
    ]


def _noisy_or(child, groups, n_parents):
    out = np.zeros((len(child), n_parents))
    for g in groups:
        out[:, g["parent_idx_super"]] = 1.0 - np.prod(1.0 - child[:, g["child_idx_sub"]], axis=1)
    return out


def _max_child(child, groups, n_parents):
    out = np.zeros((len(child), n_parents))
    for g in groups:
        out[:, g["parent_idx_super"]] = child[:, g["child_idx_sub"]].max(axis=1)
    return out


@pytest.mark.parametrize("lam", [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
def test_pruned_noisy_or_equals_max_child_synthetic(lam):
    rng = np.random.default_rng(0)
    child = rng.random((500, 6))
    groups = _synthetic_groups()
    cp = child >= lam
    pruned = prune_parent_only(cp, _noisy_or(child, groups, 3) >= lam, groups)
    assert np.array_equal(pruned, _max_child(child, groups, 3) >= lam)


def test_pruning_never_adds_a_parent_flag():
    rng = np.random.default_rng(1)
    child = rng.random((300, 6)); groups = _synthetic_groups()
    par = _noisy_or(child, groups, 3) >= 0.4
    pruned = prune_parent_only(child >= 0.4, par, groups)
    assert not (pruned & ~par).any(), "pruning must only remove flags"


def test_pruning_does_not_mutate_input():
    rng = np.random.default_rng(2)
    child = rng.random((50, 6)); groups = _synthetic_groups()
    par = (_noisy_or(child, groups, 3) >= 0.6)
    before = par.copy()
    prune_parent_only(child >= 0.6, par, groups)
    assert np.array_equal(par, before)


def test_pruning_removes_all_parent_implies_child_violations():
    rng = np.random.default_rng(3)
    child = rng.random((400, 6)); groups = _synthetic_groups()
    cp = child >= 0.5
    pruned = prune_parent_only(cp, _noisy_or(child, groups, 3) >= 0.5, groups)
    for g in groups:
        has_child = cp[:, g["child_idx_sub"]].any(axis=1)
        assert not (pruned[:, g["parent_idx_super"]] & ~has_child).any()


@needs_run
@pytest.mark.parametrize("lam", [0.1, 0.5, 0.9])
def test_pruned_noisy_or_equals_max_child_on_real_scores(lam):
    hier = build_hierarchy(verbose=False)
    groups = hier["diagnostic_groups"]
    sub = np.load(A35 / "scores_fold10.npy")[:, :23]
    pruned = prune_parent_only(sub >= lam, _noisy_or(sub, groups, 5) >= lam, groups)
    assert np.array_equal(pruned, _max_child(sub, groups, 5) >= lam)
