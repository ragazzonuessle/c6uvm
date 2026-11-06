from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from c6_uvm.ruling import (
    SplitConformal,
    empirical_coverage,
    hallucination_bound,
    marginal_band,
)


def _synthetic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    y = 1.8 * score + rng.normal(scale=0.5, size=n)
    return score.astype(np.float64), y.astype(np.float64)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(min_value=0, max_value=500))
def test_split_conformal_marginal_coverage(seed: int) -> None:
    alpha = 0.1
    s_tr, y_tr = _synthetic(300, seed)
    s_cal, y_cal = _synthetic(300, seed + 1)
    s_te, y_te = _synthetic(600, seed + 2)
    model = SplitConformal(alpha, width_budget=1e9, pe_threshold=0)
    model.fit(s_tr, y_tr, s_cal, y_cal, steps=120, lr=0.05)
    pe = np.full(s_te.shape[0], 10, dtype=np.int64)
    sets = model.predict_many(s_te, pe)
    cov = empirical_coverage(sets, y_te)
    assert cov >= 1 - alpha - 0.07


def test_qhat_grows_as_alpha_shrinks() -> None:
    s_tr, y_tr = _synthetic(400, 7)
    s_cal, y_cal = _synthetic(400, 8)
    tight = SplitConformal(0.05, 1e9, 0)
    loose = SplitConformal(0.30, 1e9, 0)
    tight.fit(s_tr, y_tr, s_cal, y_cal, 120, 0.05)
    loose.fit(s_tr, y_tr, s_cal, y_cal, 120, 0.05)
    assert tight.qhat >= loose.qhat


def test_refer_flag_fires_on_width_and_evidence() -> None:
    s_tr, y_tr = _synthetic(200, 3)
    s_cal, y_cal = _synthetic(200, 4)
    model = SplitConformal(0.1, width_budget=0.01, pe_threshold=5)
    model.fit(s_tr, y_tr, s_cal, y_cal, 80, 0.05)
    wide = model.predict(0.0, pe_count=10)
    poor = model.predict(0.0, pe_count=1)
    assert wide.refer == 1
    assert poor.refer == 1


def test_marginal_band_lower_is_one_minus_alpha() -> None:
    lo, hi = marginal_band(88, 0.1)
    assert abs(lo - 0.9) < 1e-9
    assert hi > lo


def test_hallucination_bound_monotone() -> None:
    bound = hallucination_bound(rho=0.06, eps_vlm=0.085, eta=0.18)
    assert bound.rho_plus >= bound.rho
    assert bound.holds is True
