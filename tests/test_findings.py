from __future__ import annotations

import numpy as np

from c6_uvm.findings import (
    benjamini_hochberg,
    brier_score,
    delong_test,
    dersimonian_laird,
    expected_calibration_error,
    holm_bonferroni,
    krippendorff_alpha,
    net_benefit,
)


def test_delong_separates_strong_from_weak() -> None:
    rng = np.random.default_rng(0)
    label = np.array([0] * 200 + [1] * 200, dtype=np.int64)
    strong = label + rng.normal(scale=0.3, size=400)
    weak = rng.normal(size=400)
    auc_a, auc_b, p = delong_test(label, strong, weak)
    assert auc_a > auc_b
    assert p < 0.01


def test_delong_identical_predictors_not_significant() -> None:
    rng = np.random.default_rng(1)
    label = np.array([0] * 100 + [1] * 100, dtype=np.int64)
    pred = label + rng.normal(scale=0.5, size=200)
    _, _, p = delong_test(label, pred, pred.copy())
    assert p > 0.5


def test_dersimonian_laird_zero_heterogeneity() -> None:
    est = np.array([0.80, 0.80, 0.80], dtype=np.float64)
    var = np.array([0.001, 0.001, 0.001], dtype=np.float64)
    pooled = dersimonian_laird(est, var)
    assert abs(pooled.estimate - 0.80) < 1e-6
    assert pooled.i_squared == 0.0
    assert pooled.tau_squared == 0.0


def test_calibration_metric_bounds() -> None:
    rng = np.random.default_rng(2)
    prob = rng.uniform(size=200)
    outcome = (rng.uniform(size=200) < prob).astype(np.int64)
    brier = brier_score(prob, outcome)
    ece = expected_calibration_error(prob, outcome, 10)
    assert 0.0 <= brier <= 1.0
    assert 0.0 <= ece <= 1.0


def test_net_benefit_perfect_classifier_equals_prevalence() -> None:
    rng = np.random.default_rng(3)
    outcome = (rng.uniform(size=300) < 0.4).astype(np.int64)
    prob = outcome.astype(np.float64)
    prevalence = float(outcome.mean())
    nb = net_benefit(prob, outcome, 0.25)
    assert abs(nb - prevalence) < 1e-9


def test_krippendorff_perfect_and_partial() -> None:
    perfect = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    assert abs(krippendorff_alpha(perfect) - 1.0) < 1e-9
    noisy = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]])
    assert krippendorff_alpha(noisy) < 1.0


def test_multiplicity_controls() -> None:
    pvals = np.array([0.001, 0.04, 0.20, 0.50], dtype=np.float64)
    holm = holm_bonferroni(pvals, 0.05)
    bh = benjamini_hochberg(pvals, 0.10)
    assert bool(holm[0]) is True
    assert bool(holm[3]) is False
    assert bool(bh[0]) is True
