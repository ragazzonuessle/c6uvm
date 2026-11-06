from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatVec = NDArray[np.float64]
BoolVec = NDArray[np.bool_]


def holm_bonferroni(pvalues: FloatVec, alpha: float) -> BoolVec:
    n = pvalues.shape[0]
    order = np.argsort(pvalues)
    reject = np.zeros(n, dtype=np.bool_)
    for rank, idx in enumerate(order):
        threshold = alpha / (n - rank)
        if pvalues[idx] <= threshold:
            reject[idx] = True
        else:
            break
    return reject


def benjamini_hochberg(pvalues: FloatVec, q: float) -> BoolVec:
    n = pvalues.shape[0]
    order = np.argsort(pvalues)
    reject = np.zeros(n, dtype=np.bool_)
    max_rank = -1
    for rank, idx in enumerate(order, start=1):
        if pvalues[idx] <= q * rank / n:
            max_rank = rank
    if max_rank > 0:
        for rank, idx in enumerate(order, start=1):
            if rank <= max_rank:
                reject[idx] = True
    return reject
