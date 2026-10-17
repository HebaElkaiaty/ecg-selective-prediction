"""Synthetic-data tests for the LTT threshold search.

Covers risk_control.calibrate_threshold (the validated core) and
calibrate_acuity.certify (the exact-search path that produces the primary
results). The project convention is a synthetic test for every such function,
and these are the functions every certified threshold in the paper comes from.

Risk: FNR(lambda) = P(score < lambda | y = 1), non-decreasing in lambda.
Setting lambda to the (k+1)-th smallest positive score leaves exactly k misses,
so the exact p-value there is Binom(n, alpha).cdf(k).
"""
import numpy as np
import pytest
from scipy import stats

from src.calibrate_acuity import build_candidates, certify
from src.calibration.risk_control import calibrate_threshold, validate_coverage


def fnr_at(pos, lam):
    return float((np.asarray(pos) < lam).mean())


# --------------------------------------------------------------- known answer
def test_threshold_lands_on_the_largest_tolerable_miss_count():
    """Hand-computed: Binom(100, 0.10).cdf(k) <= 0.05 holds up to k = 4."""
    n, alpha, delta = 100, 0.10, 0.05
    kmax = max(k for k in range(n + 1) if stats.binom.cdf(k, n, alpha) <= delta)
    assert kmax == 4, "premise changed; recompute the expected answer"
    pos = np.linspace(0.01, 0.99, n)                 # distinct, ascending
    lam = certify(pos, alpha, delta, build_candidates("scores", pos))
    assert lam == pytest.approx(pos[kmax])
    assert (pos < lam).sum() == kmax                 # exactly kmax misses


def test_returns_zero_when_the_floor_cannot_be_met():
    """(1-alpha)^n > delta: H0 cannot be rejected even with zero misses."""
    n, alpha, delta = 20, 0.02, 0.10 / 33
    assert (1 - alpha) ** n > delta
    pos = np.linspace(0.2, 0.9, n)
    assert certify(pos, alpha, delta, build_candidates("scores", pos)) == 0.0


def test_empty_positive_set_returns_zero():
    assert certify(np.array([]), 0.10, 0.05, np.linspace(0, 1, 10)) == 0.0


# ----------------------------------------------------------------- monotonicity
@pytest.mark.parametrize("alphas", [(0.02, 0.05, 0.10, 0.15)])
def test_threshold_is_nondecreasing_in_alpha(alphas):
    pos = np.sort(np.random.default_rng(0).random(300))
    lams = [certify(pos, a, 0.10 / 33, build_candidates("scores", pos)) for a in alphas]
    assert lams == sorted(lams), f"a looser target must not lower lambda: {lams}"


def test_threshold_is_nondecreasing_in_delta():
    pos = np.sort(np.random.default_rng(1).random(300))
    lams = [certify(pos, 0.10, d, build_candidates("scores", pos))
            for d in (0.001, 0.01, 0.05, 0.10)]
    assert lams == sorted(lams), f"a looser delta must not lower lambda: {lams}"


def test_achieved_calibration_fnr_never_exceeds_alpha():
    rng = np.random.default_rng(2)
    for _ in range(50):
        n = int(rng.integers(60, 400))
        pos = np.sort(rng.random(n))
        a = float(rng.choice([0.02, 0.05, 0.10]))
        lam = certify(pos, a, 0.10 / 33, build_candidates("scores", pos))
        if lam > 0:
            assert fnr_at(pos, lam) <= a


# ------------------------------------------------------- grid vs exact search
def test_exact_search_is_never_worse_than_a_uniform_grid():
    """The observed positive scores are an exhaustive candidate set, so the
    exact search can only find a threshold at least as large as any grid."""
    rng = np.random.default_rng(3)
    for _ in range(25):
        n = int(rng.integers(80, 400))
        pos = np.sort(rng.random(n) * 0.05)          # the regime the paper is in
        a, d = 0.10, 0.10 / 33
        assert certify(pos, a, d, build_candidates("scores", pos)) >= \
               certify(pos, a, d, build_candidates("uniform", pos)) - 1e-12


def test_certify_matches_calibrate_threshold_on_the_uniform_grid():
    """certify() must reproduce the validated core exactly when handed the
    same candidate set; only the candidates differ in the paper."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        n_pos, n_neg = int(rng.integers(50, 300)), 500
        scores = np.concatenate([rng.beta(6, 3, n_pos), rng.beta(3, 6, n_neg)])
        labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        a = float(rng.choice([0.02, 0.05, 0.10]))
        ref = calibrate_threshold(scores, labels, alpha=a, delta=0.10 / 33,
                                  grid_size=1000, bonferroni_grid=False,
                                  method="binomial")
        pos = scores[labels == 1]
        got = certify(pos, a, 0.10 / 33, build_candidates("uniform", pos))
        assert got == pytest.approx(ref, abs=1e-12)


def test_scores_candidate_set_includes_zero_and_is_sorted():
    pos = np.array([0.4, 0.1, 0.9])
    cand = build_candidates("scores", pos)
    assert cand[0] == 0.0
    assert np.all(np.diff(cand) > 0)
    assert set(pos).issubset(set(cand))


# ------------------------------------------------------------------- coverage
@pytest.mark.parametrize("alpha", [0.05, 0.10])
def test_monte_carlo_violation_rate_respects_delta(alpha):
    """The guarantee: true FNR exceeds alpha in at most delta of calibration draws."""
    r = validate_coverage(n_trials=400, n_cal=600, alpha=alpha, delta=0.10,
                          seed=7, method="binomial")
    assert r["violation_rate"] <= 0.10
    assert 0.0 < r["mean_lambda_c"] < 1.0


def test_hoeffding_is_more_conservative_than_binomial():
    """Reported in the paper: Hoeffding certifies strictly lower thresholds."""
    rng = np.random.default_rng(5)
    scores = np.concatenate([rng.beta(6, 3, 200), rng.beta(3, 6, 400)])
    labels = np.concatenate([np.ones(200), np.zeros(400)])
    kw = dict(alpha=0.10, delta=0.10, grid_size=1000, bonferroni_grid=False)
    assert calibrate_threshold(scores, labels, method="hoeffding", **kw) <= \
           calibrate_threshold(scores, labels, method="binomial", **kw)


# ------------------------------------------- the two simulations must agree
def test_both_simulations_use_the_same_estimator():
    """e4_validity_sim.certify_fast and e10's lam_for must be the same function.

    They are separate implementations of the adopted binomial monotone-boundary
    search. If they drift, the validity study and the estimator comparison would
    report different family-wise violation rates for the same procedure.
    """
    from src.revision.e10_estimator_and_dependence import lam_for
    from src.revision.e4_validity_sim import certify_fast
    rng = np.random.default_rng(11)
    for _ in range(500):
        n = int(rng.integers(5, 400))
        pos = np.sort(rng.random(n).round(int(rng.integers(2, 8))))   # induce ties
        a = float(rng.choice([0.02, 0.05, 0.10]))
        assert certify_fast(pos, a, 0.10 / 33) == pytest.approx(
            lam_for(pos, a, 0.10 / 33, "binomial", False), abs=1e-15)


def test_both_simulations_use_the_same_bootstrap_seed():
    """One canonical family-wise violation rate, not two estimates of it."""
    from src.revision.e10_estimator_and_dependence import SEED as s10
    from src.revision.e4_validity_sim import SEED as s4
    assert s10 == s4
