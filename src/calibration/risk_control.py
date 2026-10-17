"""
src/calibration/risk_control.py
--------------------------------
Single-class Learn-then-Test (LTT) calibration using the Hoeffding bound.

Reference
---------
Angelopoulos & Bates, "Learn then Test: Calibrating Predictive Algorithms to
Achieve Risk Control," 2021.  https://arxiv.org/abs/2110.01052

Risk definition
---------------
R(λ) = FNR at threshold λ = P(score < λ | label = 1).
Predicting positive iff score ≥ λ, so R(λ) is non-decreasing in λ:
a stricter threshold means more missed positives.

Algorithm
---------
For each point λ on a grid of size K in [0, 1]:
  1. Compute empirical FNR  R̂(λ) over the n_pos positive calibration samples.
  2. Hoeffding p-value for H₀: R(λ) ≥ α —
         p(λ) = exp( −2 n_pos · max(0, α − R̂(λ))² )
     n_pos is used (not n_total) because R̂(λ) is the mean of n_pos i.i.d.
     Bernoulli variables; Hoeffding requires the sample count of the average.
  3. P-value formula (method parameter):
       "binomial" (default): p(λ) = Binomial-CDF(k, n_pos, α) where k is the
         exact count of missed positives at λ.  Exact one-sided test; tight even
         for small n_pos and small α.
       "hoeffding": p(λ) = exp(−2 n_pos · max(0, α − R̂(λ))²).  Closed-form,
         distribution-free, but conservative — requires roughly 10× more positive
         calibration samples than the binomial test to certify the same threshold.
  4. Rejection threshold:
       bonferroni_grid=False (default): reject at raw δ — valid for monotone
         R(λ) because only the boundary claim matters (RCPS-style).
       bonferroni_grid=True: reject at δ/K (Bonferroni across the grid).
  5. Select λ_c = largest λ for which the test rejects.

Note: the full LTT paper combines the Hoeffding bound with the Bentkus bound
(the Hoeffding-Bentkus test, HB), which is tighter for small α. The
Hoeffding-only version here is valid and simpler; upgrading to HB is a
possible later refinement, not required for the current contribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

# Ensure repo root is importable when run as a script
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Core calibration function
# ---------------------------------------------------------------------------

def calibrate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    delta: float = 0.1,
    grid_size: int = 1000,
    bonferroni_grid: bool = False,
    method: str = "binomial",
) -> float:
    """Find the largest threshold λ_c guaranteeing FNR ≤ α at confidence 1−δ.

    Parameters
    ----------
    scores          : (n,) predicted scores ∈ [0, 1]
    labels          : (n,) binary ground-truth labels (0 or 1)
    alpha           : target FNR upper bound (e.g. 0.05)
    delta           : allowed miscoverage probability (default 0.1 → 90% guarantee)
    grid_size       : number of candidate thresholds in [0, 1] (default 1000)
    bonferroni_grid : if False (default), reject each grid point at raw δ (valid
                      for monotone R(λ)); if True, reject at δ/K (full Bonferroni).
    method          : "binomial" (default) — exact one-sided binomial p-value,
                      tight even for small n_pos; or "hoeffding" — closed-form
                      Hoeffding bound, distribution-free but conservative.

    Returns
    -------
    λ_c : float
        Calibrated threshold.  Predict positive iff score ≥ λ_c.
        Returns 0.0 if no threshold can be certified (predict all positive,
        FNR = 0 ≤ α by convention).
    """
    if method not in ("binomial", "hoeffding"):
        raise ValueError(f"method must be 'binomial' or 'hoeffding', got {method!r}")

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)

    pos_mask = labels == 1
    n_pos = int(pos_mask.sum())
    if n_pos == 0:
        return 0.0

    pos_scores = scores[pos_mask]                  # (n_pos,)
    grid = np.linspace(0.0, 1.0, grid_size)       # (K,)

    # Miss matrix: (pos_scores[:, None] < grid[None, :]) → (n_pos, K) bool.
    # Each column counts / averages how many positive-class scores fall below λ.
    # R̂(λ) is non-decreasing in λ by construction → p(λ) is non-decreasing.
    miss = (pos_scores[:, None] < grid[None, :])   # (n_pos, K)

    if method == "hoeffding":
        r_hat = miss.mean(axis=0)
        p_values = np.exp(-2.0 * n_pos * np.maximum(0.0, alpha - r_hat) ** 2)
    else:  # binomial
        k_vals = miss.sum(axis=0).astype(int)      # exact miss counts, (K,)
        # Exact one-sided p-value: P(X ≤ k | X ~ Binomial(n_pos, α))
        # Small when few misses are observed relative to what H0 predicts.
        p_values = stats.binom.cdf(k_vals, n_pos, alpha)

    # Rejection threshold: raw δ (monotone boundary) or δ/K (full Bonferroni).
    threshold = delta / grid_size if bonferroni_grid else delta

    # Exploit monotonicity: rejection set {λ : p(λ) ≤ threshold} is a contiguous
    # prefix of the grid.  searchsorted finds the boundary in O(log K), identical
    # to brute-force np.where(p_values <= threshold)[0][-1].
    cut = int(np.searchsorted(p_values, threshold, side="right")) - 1
    if cut < 0:
        return 0.0

    return float(grid[cut])


# ---------------------------------------------------------------------------
# Coverage simulation
# ---------------------------------------------------------------------------

def validate_coverage(
    n_trials: int = 1000,
    n_cal: int = 2000,
    alpha: float = 0.10,
    delta: float = 0.10,
    seed: int = 0,
    bonferroni_grid: bool = False,
    method: str = "binomial",
) -> dict:
    """Verify the FNR coverage guarantee via Monte Carlo simulation.

    True class-conditional score distributions (Beta, well-separated):
      positive class : Beta(7, 3) — mean ≈ 0.70
      negative class : Beta(3, 7) — mean ≈ 0.30

    The analytic FNR at any threshold λ is exactly CDF_Beta(7,3)(λ), so we can
    evaluate whether the calibrated λ_c truly satisfies FNR ≤ α without
    re-estimating on a fresh test set.

    For each of n_trials independent calibration draws of size n_cal:
      1. Run calibrate_threshold(scores, labels, alpha, delta).
      2. Compute true_FNR = P(score_pos < λ_c) analytically.
      3. Record a violation if true_FNR > alpha.

    The LTT guarantee: violation_rate ≤ δ with high probability.

    Parameters
    ----------
    n_trials        : number of independent calibration trials
    n_cal           : total calibration set size per trial (equal class split)
    alpha           : target FNR bound
    delta           : intended miscoverage probability (expected violation rate ≤ δ)
    seed            : base random seed
    bonferroni_grid : passed through to calibrate_threshold
    method          : passed through to calibrate_threshold

    Returns
    -------
    dict with keys:
        violation_rate  : float   — fraction of trials where true FNR > alpha
        mean_lambda_c   : float   — average calibrated threshold across trials
        mean_true_fnr   : float   — average true FNR at the calibrated threshold
    """
    pos_dist = stats.beta(7, 3)   # positive-class score distribution
    neg_dist = stats.beta(3, 7)   # negative-class score distribution

    n_pos_cal = n_cal // 2
    n_neg_cal = n_cal - n_pos_cal

    rng = np.random.default_rng(seed)
    violations = 0
    lambda_cs  = []
    true_fnrs  = []

    for _ in range(n_trials):
        pos_scores = pos_dist.rvs(n_pos_cal, random_state=rng)
        neg_scores = neg_dist.rvs(n_neg_cal, random_state=rng)
        scores = np.concatenate([pos_scores, neg_scores])
        labels = np.concatenate([np.ones(n_pos_cal), np.zeros(n_neg_cal)])

        lam = calibrate_threshold(scores, labels, alpha=alpha, delta=delta,
                                  bonferroni_grid=bonferroni_grid, method=method)

        # Analytic true FNR: P(positive-class score < λ_c) = CDF_Beta(7,3)(λ_c)
        true_fnr = float(pos_dist.cdf(lam))

        if true_fnr > alpha:
            violations += 1

        lambda_cs.append(lam)
        true_fnrs.append(true_fnr)

    return {
        "violation_rate": violations / n_trials,
        "mean_lambda_c":  float(np.mean(lambda_cs)),
        "mean_true_fnr":  float(np.mean(true_fnrs)),
    }


# ---------------------------------------------------------------------------
# CLI: run the coverage simulation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N_TRIALS = 1000
    N_CAL    = 2000
    ALPHA    = 0.10
    DELTA    = 0.10

    print("LTT Hoeffding coverage simulation — side-by-side comparison")
    print(f"  n_trials={N_TRIALS}, n_cal={N_CAL}, alpha={ALPHA}, delta={DELTA}")
    print(f"  Score distributions: positive=Beta(7,3), negative=Beta(3,7)")
    print(f"  Analytic FNR(λ) = CDF_Beta(7,3)(λ)  [exact, no re-estimation]")
    print()

    configs = [
        ("hoeffding", True,  "Hoeffding + grid-Bonferroni  (original)"),
        ("hoeffding", False, "Hoeffding + monotone-only               "),
        ("binomial",  False, "Binomial  + monotone-only  (new default)"),
    ]
    for meth, bg, label in configs:
        r = validate_coverage(n_trials=N_TRIALS, n_cal=N_CAL,
                              alpha=ALPHA, delta=DELTA,
                              bonferroni_grid=bg, method=meth)
        vr = r["violation_rate"]
        status = "PASS" if vr <= DELTA else "FAIL"
        print(f"[{label}]")
        print(f"  Violation rate : {vr:.3f}  (guarantee ≤ {DELTA})  {status}")
        print(f"  Mean λ_c       : {r['mean_lambda_c']:.4f}")
        print(f"  Mean true FNR  : {r['mean_true_fnr']:.4f}")
        print()
