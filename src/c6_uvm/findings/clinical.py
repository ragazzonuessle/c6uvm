from __future__ import annotations

from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

FloatVec = NDArray[np.float64]
IntVec = NDArray[np.int64]


def brier_score(prob: FloatVec, outcome: IntVec) -> float:
    return float(np.mean((prob - outcome) ** 2))


def expected_calibration_error(prob: FloatVec, outcome: IntVec, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    n = prob.shape[0]
    total = 0.0
    for lo, hi in pairwise(edges):
        mask = (prob >= lo) & (prob < hi) if hi < 1.0 else (prob >= lo) & (prob <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        conf = float(prob[mask].mean())
        acc = float(outcome[mask].mean())
        total += (count / n) * abs(conf - acc)
    return total


def net_benefit(prob: FloatVec, outcome: IntVec, threshold: float) -> float:
    n = prob.shape[0]
    predicted = prob >= threshold
    tp = int(np.sum(predicted & (outcome == 1)))
    fp = int(np.sum(predicted & (outcome == 0)))
    odds = threshold / (1.0 - threshold)
    return float(tp / n - (fp / n) * odds)


def net_benefit_treat_all(outcome: IntVec, threshold: float) -> float:
    prevalence = float(outcome.mean())
    odds = threshold / (1.0 - threshold)
    return float(prevalence - (1.0 - prevalence) * odds)
