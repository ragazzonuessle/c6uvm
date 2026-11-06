from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatVec = NDArray[np.float64]


class Router:
    def __init__(self, weights: dict[str, float], backbone: str) -> None:
        self._weights = dict(weights)
        self._backbone = backbone
        self._mean: dict[str, float] = {}
        self._std: dict[str, float] = {}
        self._fitted = False

    @property
    def backbone(self) -> str:
        return self._backbone

    def fit(self, scores: dict[str, FloatVec]) -> None:
        for key, vec in scores.items():
            self._mean[key] = float(vec.mean())
            self._std[key] = float(vec.std()) or 1.0
        self._fitted = True

    def _standardise(self, key: str, vec: FloatVec) -> FloatVec:
        mu = self._mean.get(key, 0.0)
        sd = self._std.get(key, 1.0)
        return (vec - mu) / sd

    def aggregate(self, scores: dict[str, FloatVec]) -> FloatVec:
        if not self._fitted:
            raise RuntimeError("router used before fit")
        keys = [k for k in self._weights if k in scores]
        total_weight = sum(self._weights[k] for k in keys)
        n = next(iter(scores.values())).shape[0]
        acc = np.zeros(n, dtype=np.float64)
        for key in keys:
            acc += self._weights[key] * self._standardise(key, scores[key])
        return acc / total_weight if total_weight else acc

    def aggregate_one(self, scores: dict[str, float]) -> float:
        keys = [k for k in self._weights if k in scores]
        total_weight = sum(self._weights[k] for k in keys)
        acc = 0.0
        for key in keys:
            sd = self._std.get(key, 1.0)
            mu = self._mean.get(key, 0.0)
            acc += self._weights[key] * (scores[key] - mu) / sd
        return acc / total_weight if total_weight else acc
