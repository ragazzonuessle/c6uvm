from __future__ import annotations

import numpy as np
from lifelines.utils import concordance_index
from numpy.typing import NDArray

FloatVec = NDArray[np.float64]
IntVec = NDArray[np.int64]


def c_index(time: FloatVec, risk: FloatVec, event: IntVec) -> float:
    return float(concordance_index(time, -risk, event))


def bootstrap_ci(
    time: FloatVec,
    risk: FloatVec,
    event: IntVec,
    draws: int,
    seed: int,
    level: float = 0.95,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = time.shape[0]
    point = c_index(time, risk, event)
    samples: list[float] = []
    for _ in range(draws):
        idx = rng.integers(0, n, size=n)
        if event[idx].sum() < 2:
            continue
        samples.append(c_index(time[idx], risk[idx], event[idx]))
    if not samples:
        return point, point, point
    lo = float(np.quantile(samples, (1.0 - level) / 2.0))
    hi = float(np.quantile(samples, 1.0 - (1.0 - level) / 2.0))
    return point, lo, hi


def paired_concordance_test(
    time: FloatVec,
    risk_a: FloatVec,
    risk_b: FloatVec,
    event: IntVec,
    draws: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    n = time.shape[0]
    deltas: list[float] = []
    for _ in range(draws):
        idx = rng.integers(0, n, size=n)
        if event[idx].sum() < 2:
            continue
        ca = c_index(time[idx], risk_a[idx], event[idx])
        cb = c_index(time[idx], risk_b[idx], event[idx])
        deltas.append(ca - cb)
    if not deltas:
        return 1.0
    arr = np.array(deltas, dtype=np.float64)
    share = float(np.mean(arr <= 0.0))
    p = 2.0 * min(share, 1.0 - share)
    return min(max(p, 1.0 / (len(arr) + 1)), 1.0)
