from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float64]


def _interval_metric(a: float, b: float) -> float:
    return (a - b) ** 2


def krippendorff_alpha(ratings: FloatMatrix) -> float:
    units = ratings.shape[1]
    per_unit: list[tuple[float, ...]] = []
    for u in range(units):
        col = ratings[:, u]
        vals = tuple(float(v) for v in col if not np.isnan(v))
        if len(vals) >= 2:
            per_unit.append(vals)
    if not per_unit:
        return float("nan")

    pooled: list[float] = []
    do_num = 0.0
    for vals in per_unit:
        m = len(vals)
        unit_sum = 0.0
        for i in range(m):
            pooled.append(vals[i])
            for j in range(m):
                if i != j:
                    unit_sum += _interval_metric(vals[i], vals[j])
        do_num += unit_sum / (m - 1)
    n_total = len(pooled)
    do = do_num / n_total

    de_num = 0.0
    for i in range(n_total):
        for j in range(n_total):
            if i != j:
                de_num += _interval_metric(pooled[i], pooled[j])
    de = de_num / (n_total * (n_total - 1)) if n_total > 1 else 0.0
    if de == 0.0:
        return 1.0
    return 1.0 - do / de
