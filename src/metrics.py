"""Calibration metrics for post-hoc ECG model evaluation.

References
----------
Guo et al. (2017) "On Calibration of Modern Neural Networks." ICML.
  https://arxiv.org/abs/1706.04599
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit  # sigmoid


def temperature_scale(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
) -> float:
    """Fit a scalar temperature T on val logits by minimising BCE loss.

    Returns T > 0; apply as `sigmoid(logits / T)` to get calibrated probs.
    Multi-label: treats each (sample, class) pair independently.

    Parameters
    ----------
    val_logits : (N, C) raw logits from the frozen model.
    val_labels : (N, C) binary ground-truth labels (0 or 1).

    Returns
    -------
    T : float  — optimal temperature (T=1 means no rescaling needed).
    """
    logits = val_logits.astype(np.float64)
    labels = val_labels.astype(np.float64)

    def bce(log_T: float) -> float:
        T = np.exp(log_T)  # unconstrained optimisation; T always > 0
        p = expit(logits / T)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    result = minimize_scalar(bce, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(result.x))


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Guo et al. 2017, Eq. (3): bins samples by confidence, measures weighted
    mean |accuracy − confidence| gap. Multi-label: all (sample, class) pairs
    are pooled into a single flat sequence before binning.

    Parameters
    ----------
    probs  : (N, C) predicted probabilities in [0, 1].
    labels : (N, C) binary ground-truth labels (0 or 1).
    n_bins : number of equal-width bins in [0, 1].

    Returns
    -------
    ece : float
    """
    p = probs.ravel().astype(np.float64)
    y = labels.ravel().astype(np.float64)
    n = len(p)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (p >= lo) & (p < hi) if lo < bin_edges[-2] else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)
