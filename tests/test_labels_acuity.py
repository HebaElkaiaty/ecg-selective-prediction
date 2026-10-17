"""
Unit tests for generate_combined_labels using synthetic metadata and
scp_statements — no real PTB-XL data.

Synthetic layout
----------------
Sub codes  : CODEA -> ClassA,  CODEB -> ClassB          (2 sub classes)
Rhythm codes: RCODE1, RCODE2                              (2 rhythm codes)
                                                 total: 4 columns

Records (meta rows)
-------------------
0  no scp codes at all
1  CODEA at 80.0  (sub ClassA only)
2  RCODE1 at 100.0 (rhythm only)
3  CODEA 60.0 + RCODE2 100.0  (one sub + one rhythm)
4  CODEA at 0.0 only  (sub at zero-likelihood — positive under universal rule)

Positivity rule (verified against PTB-XL published totals)
-----------------------------------------------------------
Universal presence-in-dict: any code that appears in scp_codes is positive
regardless of its stored likelihood value (including 0.0).  PTB-XL encodes
confirmed rhythm findings and NST_ at 0.0; both helme's STTC count and all 12
published rhythm totals match in-dict counts (within the 38-record version gap).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.labels_acuity import generate_combined_labels


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scp_statements():
    """Minimal synthetic scp_statements: 2 sub codes, 2 rhythm codes."""
    return pd.DataFrame(
        {
            "diagnostic_subclass": ["ClassA", "ClassB", np.nan,   np.nan],
            "rhythm":              [np.nan,   np.nan,   1.0,      1.0],
        },
        index=["CODEA", "CODEB", "RCODE1", "RCODE2"],
    )


@pytest.fixture
def meta():
    """5-row synthetic metadata with hand-crafted scp_codes dicts."""
    return pd.DataFrame(
        {
            "scp_codes": [
                {},                                          # row 0: no codes
                {"CODEA": 80.0},                            # row 1: sub only
                {"RCODE1": 100.0},                          # row 2: rhythm only
                {"CODEA": 60.0, "RCODE2": 100.0},          # row 3: sub + rhythm
                {"CODEA": 0.0},                              # row 4: sub code at zero-likelihood → sub col 0
            ]
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_rhythm_code_gives_all_zero_rhythm_segment(meta, scp_statements):
    """Row 1 has only a sub code — its rhythm slice must be all zeros."""
    matrix, class_names = generate_combined_labels(meta, scp_statements)
    n_sub = sum(1 for n in class_names if n in ["ClassA", "ClassB"])
    rhythm_slice = matrix[1, n_sub:]
    assert rhythm_slice.sum() == 0.0, (
        f"Expected all-zero rhythm segment for row 1, got {rhythm_slice}"
    )


def test_known_sub_code_gives_correct_one(meta, scp_statements):
    """Row 1 has CODEA->ClassA at likelihood 80. ClassA column must be 1, ClassB must be 0."""
    matrix, class_names = generate_combined_labels(meta, scp_statements)
    sub_names = [n for n in class_names if n in ["ClassA", "ClassB"]]
    idx_A = class_names.index("ClassA")
    idx_B = class_names.index("ClassB")
    assert matrix[1, idx_A] == 1.0, "ClassA should be 1 for row 1"
    assert matrix[1, idx_B] == 0.0, "ClassB should be 0 for row 1"


def test_sub_code_at_zero_likelihood_is_positive(meta, scp_statements):
    """Row 4 has CODEA=0.0 — under the universal presence rule, ClassA must be 1."""
    matrix, class_names = generate_combined_labels(meta, scp_statements)
    idx_A = class_names.index("ClassA")
    idx_B = class_names.index("ClassB")
    assert matrix[4, idx_A] == 1.0, "ClassA=0.0 should be positive (universal presence rule)"
    assert matrix[4, idx_B] == 0.0, "ClassB absent should be 0"


def test_combined_column_count_equals_sub_plus_rhythm(meta, scp_statements):
    """Matrix columns == number of distinct sub labels + number of rhythm codes."""
    matrix, class_names = generate_combined_labels(meta, scp_statements)
    n_sub_labels = 2    # ClassA, ClassB
    n_rhy_labels = 2    # RCODE1, RCODE2
    assert matrix.shape[1] == n_sub_labels + n_rhy_labels, (
        f"Expected {n_sub_labels + n_rhy_labels} columns, got {matrix.shape[1]}"
    )
    assert len(class_names) == matrix.shape[1]


def test_row_count_matches_metadata(meta, scp_statements):
    """Output matrix must have exactly len(meta) rows."""
    matrix, class_names = generate_combined_labels(meta, scp_statements)
    assert matrix.shape[0] == len(meta), (
        f"Expected {len(meta)} rows, got {matrix.shape[0]}"
    )


def test_sub_columns_precede_rhythm_columns(meta, scp_statements):
    """Subdiagnostic names (ClassA, ClassB) must appear before rhythm names in class_names."""
    _, class_names = generate_combined_labels(meta, scp_statements)
    sub_indices = [i for i, n in enumerate(class_names) if n in {"ClassA", "ClassB"}]
    rhy_indices = [i for i, n in enumerate(class_names) if n in {"RCODE1", "RCODE2"}]
    assert max(sub_indices) < min(rhy_indices), (
        f"Sub indices {sub_indices} should all precede rhythm indices {rhy_indices}"
    )


def test_combined_row_with_sub_and_rhythm(meta, scp_statements):
    """Row 3 has CODEA->ClassA (sub) and RCODE2 (rhythm) — both slots must be 1."""
    matrix, class_names = generate_combined_labels(meta, scp_statements)
    idx_A     = class_names.index("ClassA")
    idx_rcode2 = class_names.index("RCODE2")
    assert matrix[3, idx_A]      == 1.0, "ClassA should be 1 for row 3"
    assert matrix[3, idx_rcode2] == 1.0, "RCODE2 should be 1 for row 3"


def test_output_dtype_is_float32(meta, scp_statements):
    matrix, _ = generate_combined_labels(meta, scp_statements)
    assert matrix.dtype == np.float32


# ---------------------------------------------------------------------------
# Split-positivity rule: zero-threshold rhythm codes (SR etc.) vs standard
# ---------------------------------------------------------------------------

@pytest.fixture
def scp_statements_sr():
    """scp_statements with SR (zero-threshold), AFIB and AFLT (standard threshold)."""
    return pd.DataFrame(
        {
            "diagnostic_subclass": [np.nan,  np.nan,  np.nan],
            "rhythm":              [1.0,     1.0,     1.0],
        },
        index=["SR", "AFIB", "AFLT"],
    )


@pytest.fixture
def meta_sr():
    """3-row metadata for split-positivity tests."""
    return pd.DataFrame(
        {
            "scp_codes": [
                {"SR": 0.0},                     # row 0: SR present at zero-likelihood
                {"AFIB": 0.0, "AFLT": 100.0},   # row 1: AFIB at zero, AFLT at 100
                {"AFIB": 100.0},                 # row 2: AFIB at 100
            ]
        }
    )


def test_sr_at_zero_likelihood_is_positive(meta_sr, scp_statements_sr):
    """SR=0.0 must register as positive (zero-threshold code)."""
    matrix, class_names = generate_combined_labels(meta_sr, scp_statements_sr)
    sr_idx = class_names.index("SR")
    assert matrix[0, sr_idx] == 1.0, "SR=0.0 should be positive"
    # Sub segment (empty for this fixture) and other rhythm columns must be 0
    other = [i for i, n in enumerate(class_names) if n != "SR"]
    assert matrix[0, other].sum() == 0.0, "Only SR should be 1 for row 0"


def test_afib_zero_and_aflt_100_both_positive(meta_sr, scp_statements_sr):
    """Row 1: AFIB=0.0 and AFLT=100.0 — both positive under universal presence rule."""
    matrix, class_names = generate_combined_labels(meta_sr, scp_statements_sr)
    afib_idx = class_names.index("AFIB")
    aflt_idx = class_names.index("AFLT")
    assert matrix[1, afib_idx] == 1.0, "AFIB=0.0 should be positive (rhythm presence rule)"
    assert matrix[1, aflt_idx] == 1.0, "AFLT=100.0 should be positive"


def test_afib_at_100_is_positive(meta_sr, scp_statements_sr):
    """AFIB=100.0 must be positive."""
    matrix, class_names = generate_combined_labels(meta_sr, scp_statements_sr)
    afib_idx = class_names.index("AFIB")
    assert matrix[2, afib_idx] == 1.0, "AFIB=100.0 should be positive"


def test_generic_rhythm_code_at_zero_likelihood_is_positive(scp_statements):
    """Any rhythm code at 0.0 must be positive — universal presence rule applies to all 12."""
    meta_rcode_zero = pd.DataFrame({"scp_codes": [{"RCODE1": 0.0}]})
    matrix, class_names = generate_combined_labels(meta_rcode_zero, scp_statements)
    rcode1_idx = class_names.index("RCODE1")
    assert matrix[0, rcode1_idx] == 1.0, "Rhythm code at 0.0 should be positive"
