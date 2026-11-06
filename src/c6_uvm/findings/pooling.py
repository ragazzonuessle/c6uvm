from __future__ import annotations

import numpy as np
from attrs import frozen
from numpy.typing import NDArray

FloatVec = NDArray[np.float64]


@frozen
class PooledEstimate:
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    q: float
    i_squared: float
    tau_squared: float


def dersimonian_laird(estimates: FloatVec, variances: FloatVec) -> PooledEstimate:
    fixed_w = 1.0 / variances
    fixed_mean = float(np.sum(fixed_w * estimates) / np.sum(fixed_w))
    q = float(np.sum(fixed_w * (estimates - fixed_mean) ** 2))
    k = estimates.shape[0]
    df = k - 1
    c = float(np.sum(fixed_w) - np.sum(fixed_w**2) / np.sum(fixed_w))
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    rand_w = 1.0 / (variances + tau2)
    mean = float(np.sum(rand_w * estimates) / np.sum(rand_w))
    se = float(np.sqrt(1.0 / np.sum(rand_w)))
    i2 = max(0.0, (q - df) / q) * 100.0 if q > 0 else 0.0
    return PooledEstimate(
        estimate=mean,
        se=se,
        ci_low=mean - 1.96 * se,
        ci_high=mean + 1.96 * se,
        q=q,
        i_squared=i2,
        tau_squared=tau2,
    )
