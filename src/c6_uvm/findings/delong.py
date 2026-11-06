from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatVec = NDArray[np.float64]
IntVec = NDArray[np.int64]


def _midrank(x: FloatVec) -> FloatVec:
    order = np.argsort(x)
    ranked = x[order]
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j] == ranked[i]:
            j += 1
        out[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    result = np.empty(n, dtype=np.float64)
    result[order] = out
    return result


def _structural_components(predictions: FloatVec, m: int) -> tuple[FloatVec, FloatVec, FloatVec]:
    k = predictions.shape[0]
    n = predictions.shape[1] - m
    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)
    for r in range(k):
        pos = predictions[r, :m]
        neg = predictions[r, m:]
        tx[r] = _midrank(pos)
        ty[r] = _midrank(neg)
        tz[r] = _midrank(predictions[r])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1.0) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    return aucs, v01, v10


def _cov(v01: FloatVec, v10: FloatVec, m: int, n: int) -> FloatVec:
    s01 = np.cov(v01)
    s10 = np.cov(v10)
    return np.atleast_2d(s01) / m + np.atleast_2d(s10) / n


def delong_test(label: IntVec, pred_a: FloatVec, pred_b: FloatVec) -> tuple[float, float, float]:
    order = np.argsort(-label)
    label_sorted = label[order]
    m = int(label_sorted.sum())
    preds = np.vstack([pred_a[order], pred_b[order]])
    aucs, v01, v10 = _structural_components(preds, m)
    n = preds.shape[1] - m
    cov = _cov(v01, v10, m, n)
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    if var <= 0.0:
        p = 1.0 if aucs[0] == aucs[1] else 0.0
        return float(aucs[0]), float(aucs[1]), p
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = float(2.0 * stats.norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), p
