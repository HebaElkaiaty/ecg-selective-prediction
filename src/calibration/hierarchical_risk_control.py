"""
src/calibration/hierarchical_risk_control.py
---------------------------------------------
Contribution 2: hierarchy-constrained conformal prediction sets.

Single-model design
-------------------
Only sigmoid(acuity_val/test_logits.npy) is used — the 35-class model.

Joint probability matrix (N, 28):
  cols  0-22 : subdiagnostic scores from the 35-class model output
  cols 23-27 : superdiagnostic scores derived via noisy-OR aggregation over
               the group's children, using the scp_statements.csv-derived mapping.
               Order: NORM=23, MI=24, STTC=25, CD=26, HYP=27.

NORM group: NORM's only subdiagnostic child is NORM itself (index 14).
noisy-OR over a single input equals that input, so the NORM parent score
is identical to the NORM subdiag score. This means the NORM group is
always internally consistent by construction (both scores are identical
and both are above or below threshold together). Handled explicitly here.

Calibration
-----------
A single global threshold λ ∈ [0,1] is applied to all 28 columns.
The inconsistency loss ℓ(x, λ) = 1 if S(x, λ) violates any hierarchy
constraint, 0 otherwise. The risk R(λ) = E[ℓ(x, λ)] is non-monotone in λ
(zero at both extremes, peaks in the interior). Grid-wide Bonferroni
(bonferroni_grid=True) is therefore mandatory.

Interface note on calibrate_threshold() from risk_control.py
-------------------------------------------------------------
calibrate_threshold() takes a fixed per-record score and a threshold to search
over. Our loss ℓ(x_i, λ) depends on the global threshold λ being searched —
there is no fixed score_i such that ℓ_i(λ) = 1{score_i < λ} for all λ.
A thin wrapper (calibrate_inconsistency_threshold) implements the identical
LTT binomial math (same scipy.stats.binom.cdf call, same Bonferroni structure)
without modifying the core function.

References
----------
Angelopoulos & Bates, "Learn then Test," 2021.
Bates et al., "Distribution-Free Risk-Controlling Prediction Sets," JACM 2021.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# noisy-OR score construction
# ---------------------------------------------------------------------------

# Joint-matrix layout: cols 0-22 subdiagnostic, 23-27 noisy-OR parents.
_NORM_SUBDIAG_COL = 14
_NORM_PARENT_COL  = 23


def build_noisy_or_joint(
    probs35: np.ndarray,
    diagnostic_groups: list[dict],
) -> np.ndarray:
    """Build the (N, 28) joint probability matrix using noisy-OR aggregation.

    Parameters
    ----------
    probs35          : (N, 35) sigmoid probabilities from the 35-class model
    diagnostic_groups: from build_hierarchy()["diagnostic_groups"] — order must
                       be NORM, MI, STTC, CD, HYP to match the col-23..27 layout

    Returns
    -------
    joint : (N, 28)
        Cols  0-22 : subdiagnostic scores (probs35[:, :23])
        Cols 23-27 : noisy-OR parent scores in diagnostic_groups order

    NORM handling
    -------------
    NORM's only child in our 23-class subdiag set is NORM itself (index 14).
    noisy-OR over a single value p equals p. So col 23 (NORM parent) = col 14
    (NORM subdiag). The NORM group is therefore always self-consistent at any
    threshold — both columns carry the same score. This is noted explicitly
    rather than assumed.
    """
    sub = probs35[:, :23].copy()   # (N, 23)
    N   = sub.shape[0]
    G   = len(diagnostic_groups)

    super_scores = np.empty((N, G), dtype=float)
    for i, grp in enumerate(diagnostic_groups):
        child_probs = sub[:, grp["child_idx_sub"]]      # (N, n_children)
        # noisy-OR: 1 − ∏(1 − p_k) over children
        super_scores[:, i] = 1.0 - np.prod(1.0 - child_probs, axis=1)

    return np.concatenate([sub, super_scores], axis=1)  # (N, 28)


# ---------------------------------------------------------------------------
# Per-record inconsistency computation (used by calibration and tests)
# ---------------------------------------------------------------------------

def compute_joint_inconsistency(
    joint_probs: np.ndarray,
    thresholds: np.ndarray,
    groups: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-record inconsistency indicators for a single threshold vector.

    Parameters
    ----------
    joint_probs : (N, C) — joint probability matrix (C=28 in production;
                  any width in tests, as long as group col indices are valid)
    thresholds  : (C,)   — per-class thresholds, clipped to [0, 1] internally
    groups      : list of dicts with ABSOLUTE joint-matrix column indices:
                    parent_col : int       — col for the parent class
                    child_cols : list[int] — cols for the children
                  Build with make_groups() from build_hierarchy() output.

    Returns
    -------
    inconsistent : (N,) bool — True if record violates any group constraint
    per_group    : (N, G) bool — per-group violation flags
    """
    thresholds = np.clip(np.asarray(thresholds, dtype=float), 0.0, 1.0)
    N = joint_probs.shape[0]
    G = len(groups)

    pred      = joint_probs >= thresholds[np.newaxis, :]   # (N, C)
    per_group = np.zeros((N, G), dtype=bool)

    for g_idx, grp in enumerate(groups):
        parent_pred = pred[:, grp["parent_col"]]
        any_child   = pred[:, grp["child_cols"]].any(axis=1)

        per_group[:, g_idx] = (any_child & ~parent_pred) | (parent_pred & ~any_child)

    return per_group.any(axis=1), per_group


def make_groups(diagnostic_groups: list[dict], n_sub: int = 23) -> list[dict]:
    """Convert build_hierarchy() groups to absolute joint-matrix column indices.

    child_cols  = child_idx_sub  (subdiag block, already absolute 0..n_sub-1)
    parent_col  = n_sub + parent_idx_super  (superdiag block starts at n_sub)
    """
    return [
        {
            "parent_col": n_sub + grp["parent_idx_super"],
            "child_cols": grp["child_idx_sub"],
        }
        for grp in diagnostic_groups
    ]


# ---------------------------------------------------------------------------
# Calibration wrapper (thin LTT binomial wrapper for inconsistency loss)
# ---------------------------------------------------------------------------

def calibrate_inconsistency_threshold(
    val_joint: np.ndarray,
    groups: list[dict],
    alpha: float = 0.05,
    delta: float = 0.10,
    grid_size: int = 1000,
) -> dict:
    """Find the λ closest to 0.5 that certifies inconsistency rate ≤ α.

    Implements the same LTT binomial test used in calibrate_threshold()
    (risk_control.py) but for a loss that depends on λ — so calibrate_threshold()
    cannot be called directly.  The math is identical: for each grid point j,
      k_j   = number of inconsistent val records at threshold grid[j]
      p_j   = Binom.CDF(k_j, N, α)                 ← same scipy call
      reject iff p_j ≤ δ/grid_size                  ← grid-wide Bonferroni

    bonferroni_grid=True is mandatory because R(λ) is non-monotone.

    Returns the certified λ closest to 0.5 (minimum perturbation from the
    natural decision boundary) among all certified grid points. If multiple
    disconnected certified regions exist, this picks the one nearest 0.5.

    Parameters
    ----------
    val_joint : (N, 28) joint probability matrix on the val fold
    groups    : from make_groups() — absolute joint-matrix column indices
    alpha     : target inconsistency rate upper bound
    delta     : allowed miscoverage probability (family-wise over the grid)
    grid_size : number of candidate thresholds in [0, 1]

    Returns
    -------
    dict with keys:
        certified          : bool
        no_useful_certificate : bool -- True when nothing certifies, when the
                             only certified threshold is lambda=0 (predict every
                             class, trivially consistent), or when >= 99% of
                             calibration records receive an empty prediction set
                             at lambda_c. A certificate can be valid and still
                             describe no usable operating point; this flag makes
                             that explicit rather than leaving it to be inferred
                             from the set-size column.
        empty_set_rate_at_lc : float -- fraction of calibration records with an
                             empty corrected set at lambda_c (NaN if uncertified)
        lambda_c           : float  (NaN if uncertified)
        val_incons_at_half : float  — inconsistency rate at λ=0.5 (baseline)
        val_incons_at_lc   : float  — inconsistency rate at λ_c (NaN if uncertified)
        n_certified        : int    — number of certified grid points
        certified_range    : (min, max) certified λ values (NaN tuple if uncertified)
        grid               : (grid_size,) ndarray
        incons_rates       : (grid_size,) ndarray — R(λ) for each grid point
        p_values           : (grid_size,) ndarray — binomial p-values
    """
    N    = val_joint.shape[0]
    C    = val_joint.shape[1]
    grid = np.linspace(0.0, 1.0, grid_size)   # (K,)

    def _empty_rate(lam: float) -> float:
        """Fraction of records whose corrected prediction set is empty at lam.

        The NORM pair (subdiag col 14, parent col 23) carries identical scores
        under noisy-OR over a single child, so it counts as one finding.
        """
        pred = val_joint >= lam
        size = pred.sum(axis=1)
        if C > _NORM_PARENT_COL:
            size = size - (pred[:, _NORM_SUBDIAG_COL] & pred[:, _NORM_PARENT_COL])
        return float((size == 0).mean())

    # Vectorised inconsistency over all grid thresholds in one pass.
    # Broadcast: (N, C, 1) >= (1, 1, K) → (N, C, K)
    pred_all = (val_joint[:, :, np.newaxis]
                >= grid[np.newaxis, np.newaxis, :])   # (N, C, K) bool

    incons_all = np.zeros((N, grid_size), dtype=bool)
    for grp in groups:
        parent_pred = pred_all[:, grp["parent_col"], :]               # (N, K)
        any_child   = pred_all[:, grp["child_cols"], :].any(axis=1)  # (N, K)
        incons_all |= (any_child & ~parent_pred) | (parent_pred & ~any_child)

    k_vals      = incons_all.sum(axis=0).astype(int)   # (K,)
    incons_rates = k_vals / N
    p_values    = stats.binom.cdf(k_vals, N, alpha)    # (K,)

    rejection_threshold = delta / grid_size             # grid-wide Bonferroni
    certified_mask      = p_values <= rejection_threshold

    n_certified = int(certified_mask.sum())

    # Baseline at λ=0.5
    idx_half           = int(np.argmin(np.abs(grid - 0.5)))
    val_incons_at_half = float(incons_rates[idx_half])

    if n_certified == 0:
        return {
            "certified":          False,
            "no_useful_certificate": True,
            "empty_set_rate_at_lc": float("nan"),
            "lambda_c":           float("nan"),
            "val_incons_at_half": val_incons_at_half,
            "val_incons_at_lc":   float("nan"),
            "n_certified":        0,
            "certified_range":    (float("nan"), float("nan")),
            "grid":               grid,
            "incons_rates":       incons_rates,
            "p_values":           p_values,
        }

    certified_lambdas = grid[certified_mask]
    idx_best          = int(np.argmin(np.abs(certified_lambdas - 0.5)))
    lambda_c          = float(certified_lambdas[idx_best])
    val_incons_at_lc  = float(incons_rates[np.argmin(np.abs(grid - lambda_c))])

    empty_rate = _empty_rate(lambda_c)
    return {
        "certified":          True,
        "no_useful_certificate": bool(lambda_c <= 0.0 or empty_rate >= 0.99),
        "empty_set_rate_at_lc": empty_rate,
        "lambda_c":           lambda_c,
        "val_incons_at_half": val_incons_at_half,
        "val_incons_at_lc":   val_incons_at_lc,
        "n_certified":        n_certified,
        "certified_range":    (float(certified_lambdas.min()),
                               float(certified_lambdas.max())),
        "grid":               grid,
        "incons_rates":       incons_rates,
        "p_values":           p_values,
    }


# ---------------------------------------------------------------------------
# Monte Carlo coverage validation
# ---------------------------------------------------------------------------

def validate_consistency_coverage(
    n_trials: int = 500,
    n_cal: int = 1000,
    n_test: int = 5000,
    n_groups: int = 5,
    alpha: float = 0.05,
    delta: float = 0.10,
    grid_size: int = 1000,
    seed: int = 42,
) -> dict:
    """Verify the inconsistency-rate guarantee via Monte Carlo simulation.

    Synthetic setup: 2 children per group, noisy-OR parent.
    Child scores ~ Beta(5, 2); independently drawn (no cross-group correlation).
    True inconsistency at the certified λ is estimated empirically on n_test
    held-out records. A trial violates if estimated test inconsistency > alpha.

    Returns
    -------
    dict: violation_rate, mean_lambda_c, mean_test_incons
    """
    rng            = np.random.default_rng(seed)
    n_child        = 2
    n_sub          = n_groups * n_child
    n_joint        = n_sub + n_groups

    # Groups with absolute joint-col indices
    groups = [
        {
            "parent_col": n_sub + g,
            "child_cols": [g * n_child, g * n_child + 1],
        }
        for g in range(n_groups)
    ]

    violations  = 0
    lambda_cs   = []
    test_incons = []

    for _ in range(n_trials):
        # Calibration set: child scores from Beta(5,2); parents via noisy-OR
        child_cal   = rng.beta(5, 2, (n_cal, n_sub))
        parent_cal  = np.column_stack([
            1.0 - np.prod(1.0 - child_cal[:, g*n_child:(g+1)*n_child], axis=1)
            for g in range(n_groups)
        ])
        cal_joint   = np.concatenate([child_cal, parent_cal], axis=1)

        result = calibrate_inconsistency_threshold(
            cal_joint, groups,
            alpha=alpha, delta=delta, grid_size=grid_size,
        )

        if not result["certified"]:
            lambda_cs.append(float("nan"))
            test_incons.append(float("nan"))
            continue

        lam = result["lambda_c"]
        lambda_cs.append(lam)

        # Test set
        child_test  = rng.beta(5, 2, (n_test, n_sub))
        parent_test = np.column_stack([
            1.0 - np.prod(1.0 - child_test[:, g*n_child:(g+1)*n_child], axis=1)
            for g in range(n_groups)
        ])
        test_joint  = np.concatenate([child_test, parent_test], axis=1)

        thresholds  = np.full(n_joint, lam)
        incons_t, _ = compute_joint_inconsistency(test_joint, thresholds, groups)
        true_incons = float(incons_t.mean())
        test_incons.append(true_incons)

        if true_incons > alpha:
            violations += 1

    valid = [x for x in test_incons if not np.isnan(x)]
    return {
        "violation_rate":   violations / n_trials,
        "mean_lambda_c":    float(np.nanmean(lambda_cs)),
        "mean_test_incons": float(np.mean(valid)) if valid else float("nan"),
    }
