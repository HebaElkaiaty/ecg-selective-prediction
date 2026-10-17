"""Synthetic-data unit tests for src/metrics.py."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics import temperature_scale, expected_calibration_error


# ── temperature_scale ──────────────────────────────────────────────────────

def test_temperature_scale_returns_valid_T():
    """temperature_scale returns a finite positive scalar regardless of input."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, (500, 5)).astype(np.float32)
    logits = rng.normal(0, 2.0, (500, 5)).astype(np.float32)
    T = temperature_scale(logits, labels)
    assert np.isfinite(T) and T > 0.0, f"T not a valid positive scalar: {T}"


def test_temperature_scale_overconfident_random_labels():
    """Large-magnitude logits on random labels → T >> 1 (squish back toward 0.5).

    When the model is confidently wrong half the time, optimal T is large.
    Logit sign is randomised independently of labels so predictions are noise.
    """
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, (400, 5)).astype(np.float32)
    # logit signs drawn independently of labels → model is uninformative but confident
    signs  = rng.choice([-1.0, 1.0], size=(400, 5))
    logits = (signs * 10.0).astype(np.float32)
    T = temperature_scale(logits, labels)
    assert T > 1.5, f"Expected T >> 1 for overconfident uninformative model, got {T:.4f}"


def test_temperature_scale_underconfident():
    """Very small logits (underconfident) → T should be < 1."""
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 2, (400, 5)).astype(np.float32)
    # logits near zero: underconfident
    logits = np.where(labels == 1, 0.1, -0.1).astype(np.float32)
    T = temperature_scale(logits, labels)
    assert T < 0.8, f"Expected T < 1 for underconfident model, got {T:.4f}"


# ── expected_calibration_error ─────────────────────────────────────────────

def test_ece_perfect_calibration():
    """Probabilities exactly equal empirical frequencies → ECE ≈ 0."""
    # construct bins with exact match: prob=0.2 and accuracy=0.2, etc.
    n = 1000
    probs = np.full((n, 1), 0.2)
    rng = np.random.default_rng(42)
    labels = (rng.random((n, 1)) < 0.2).astype(np.float32)
    ece = expected_calibration_error(probs, labels)
    # statistical noise but should be small
    assert ece < 0.05, f"ECE should be near 0 for calibrated probs, got {ece:.4f}"


def test_ece_worst_case():
    """Probs = 1.0 but labels = 0 everywhere → ECE = 1.0."""
    probs  = np.ones((100, 3))
    labels = np.zeros((100, 3))
    ece = expected_calibration_error(probs, labels)
    assert abs(ece - 1.0) < 1e-6, f"Expected ECE=1.0, got {ece:.6f}"


def test_ece_returns_scalar_in_0_1():
    rng = np.random.default_rng(7)
    probs  = rng.random((200, 5))
    labels = rng.integers(0, 2, (200, 5)).astype(float)
    ece = expected_calibration_error(probs, labels)
    assert 0.0 <= ece <= 1.0, f"ECE out of range: {ece}"


def test_temperature_improves_ece():
    """Applying the fitted temperature should not worsen ECE."""
    from scipy.special import expit
    rng = np.random.default_rng(99)
    labels = rng.integers(0, 2, (600, 5)).astype(np.float32)
    logits = np.where(labels == 1, 15.0, -15.0).astype(np.float32)

    T = temperature_scale(logits, labels)
    probs_raw   = expit(logits).astype(np.float32)
    probs_cal   = expit(logits / T).astype(np.float32)

    ece_before = expected_calibration_error(probs_raw, labels)
    ece_after  = expected_calibration_error(probs_cal, labels)

    assert ece_after < ece_before, (
        f"Temperature scaling should improve ECE: before={ece_before:.4f}, "
        f"after={ece_after:.4f}, T={T:.4f}"
    )


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v"]))
