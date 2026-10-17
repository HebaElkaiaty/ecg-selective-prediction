"""
tests/test_hierarchy.py
-----------------------
Unit tests for hierarchy-constrained inconsistency detection.

Synthetic 3-class hierarchy: 1 parent (col 1 in superdiag block),
2 children (cols 0-1 in subdiag block).  Joint matrix is (N, 3):
  joint[:, 0]  — child 0   (subdiag col 0)
  joint[:, 1]  — child 1   (subdiag col 1)
  joint[:, 2]  — parent P  (superdiag col 0, n_sub=2 so joint col 2)

Groups list for compute_joint_inconsistency:
  parent_idx_super = 0   (col 0 within the superdiag block of size 1)
  child_idx_sub    = [0, 1]

Threshold = 0.5 for all classes throughout.

Hand-verified truth table:
  C0=1, C1=0, P=1 → consistent (one child + parent present)
  C0=0, C1=0, P=0 → consistent (empty set — nothing to be inconsistent about)
  C0=1, C1=0, P=0 → INCONSISTENT (child predicted, parent absent)
  C0=0, C1=0, P=1 → INCONSISTENT (parent predicted, no child present)
  C0=1, C1=1, P=1 → consistent (all three present)
  C0=1, C1=1, P=0 → INCONSISTENT (two children predicted, parent absent)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration.hierarchical_risk_control import compute_joint_inconsistency

# Single group: joint matrix has 3 cols — [child0, child1, parent]
# Use absolute column indices (parent_col, child_cols) as required by the function.
_GROUPS = [{"parent_col": 2, "child_cols": [0, 1]}]
_THRESH = np.array([0.5, 0.5, 0.5])  # threshold for [child0, child1, parent]

# Helper: build a joint_probs row from binary on/off predictions.
# Uses 0.9 for "on" (above 0.5) and 0.1 for "off" (below 0.5).
def _row(c0: int, c1: int, p: int) -> np.ndarray:
    vals = [0.9 if c0 else 0.1, 0.9 if c1 else 0.1, 0.9 if p else 0.1]
    return np.array(vals, dtype=float)[np.newaxis, :]  # (1, 3)


def test_both_absent_consistent():
    """Empty set: no predictions → trivially consistent."""
    incons, pg = compute_joint_inconsistency(_row(0, 0, 0), _THRESH, _GROUPS)
    assert not incons[0], "Empty prediction set must be consistent"
    assert not pg[0, 0]


def test_child_and_parent_both_present():
    """One child + parent: satisfies both directions."""
    incons, pg = compute_joint_inconsistency(_row(1, 0, 1), _THRESH, _GROUPS)
    assert not incons[0], "child+parent prediction should be consistent"


def test_all_present_consistent():
    """All three present: both children + parent."""
    incons, pg = compute_joint_inconsistency(_row(1, 1, 1), _THRESH, _GROUPS)
    assert not incons[0], "All three present should be consistent"


def test_child_without_parent_inconsistent():
    """Child predicted but parent absent: child-implies-parent violated."""
    incons, pg = compute_joint_inconsistency(_row(1, 0, 0), _THRESH, _GROUPS)
    assert incons[0], "Child without parent must be inconsistent"
    assert pg[0, 0]


def test_second_child_without_parent_inconsistent():
    """Second child predicted, parent absent."""
    incons, pg = compute_joint_inconsistency(_row(0, 1, 0), _THRESH, _GROUPS)
    assert incons[0], "Second child without parent must be inconsistent"


def test_both_children_without_parent_inconsistent():
    """Both children predicted but parent absent."""
    incons, pg = compute_joint_inconsistency(_row(1, 1, 0), _THRESH, _GROUPS)
    assert incons[0], "Two children without parent must be inconsistent"


def test_parent_without_child_inconsistent():
    """Parent predicted but no children: parent-implies-child violated."""
    incons, pg = compute_joint_inconsistency(_row(0, 0, 1), _THRESH, _GROUPS)
    assert incons[0], "Parent without child must be inconsistent"
    assert pg[0, 0]


def test_batch_mixed():
    """Batch of 4 records with mixed consistency."""
    joint = np.array([
        [0.9, 0.1, 0.9],   # child0+parent — consistent
        [0.9, 0.1, 0.1],   # child0, no parent — inconsistent
        [0.1, 0.1, 0.9],   # parent, no child — inconsistent
        [0.1, 0.1, 0.1],   # empty — consistent
    ])
    incons, pg = compute_joint_inconsistency(joint, _THRESH, _GROUPS)
    expected = [False, True, True, False]
    assert list(incons) == expected, f"Expected {expected}, got {list(incons)}"


def test_two_independent_groups():
    """Two groups, inconsistency in second group detected while first is clean."""
    # joint cols: [child0_g0, child1_g0, child0_g1, child1_g1, parent_g0, parent_g1]
    groups_2 = [
        {"parent_col": 4, "child_cols": [0, 1]},
        {"parent_col": 5, "child_cols": [2, 3]},
    ]
    thresh_2 = np.full(6, 0.5)

    # Record: group0 consistent (child0+parent0), group1 inconsistent (parent1, no child)
    joint = np.array([[0.9, 0.1, 0.1, 0.1, 0.9, 0.9]])  # (1, 6)
    incons, pg = compute_joint_inconsistency(joint, thresh_2, groups_2)
    assert incons[0], "Record should be inconsistent due to group 1"
    assert not pg[0, 0], "Group 0 should be consistent"
    assert pg[0, 1],     "Group 1 should be inconsistent"


def test_threshold_boundary():
    """Scores exactly at threshold: >= threshold is predicted positive."""
    # score = 0.5 exactly = threshold 0.5 → predicted (>= is inclusive)
    joint = np.array([[0.5, 0.1, 0.1]])   # child0 at boundary, no parent
    incons, _ = compute_joint_inconsistency(joint, _THRESH, _GROUPS)
    assert incons[0], "Score == threshold should be positive → inconsistent"
