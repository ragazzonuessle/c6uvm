from __future__ import annotations

import numpy as np
import torch
from attrs import frozen
from numpy.typing import NDArray

FloatVec = NDArray[np.float64]


@frozen
class QuantileLine:
    slope: float
    intercept: float

    def __call__(self, score: FloatVec) -> FloatVec:
        return (self.slope * score + self.intercept).astype(np.float64)


@frozen
class PredictionSet:
    lower: float
    upper: float
    refer: int

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, y: float) -> bool:
        return self.lower <= y <= self.upper


def _pinball(residual: torch.Tensor, q: float) -> torch.Tensor:
    return torch.maximum(q * residual, (q - 1.0) * residual).mean()


def fit_quantile_line(
    score: FloatVec, y: FloatVec, q: float, steps: int, lr: float
) -> QuantileLine:
    xs = torch.as_tensor(score, dtype=torch.float32)
    ys = torch.as_tensor(y, dtype=torch.float32)
    a = torch.zeros(1, requires_grad=True)
    b = torch.tensor([float(np.quantile(y, q))], requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = a * xs + b
        loss = _pinball(ys - pred, q)
        torch.autograd.backward(loss)
        opt.step()
    return QuantileLine(slope=float(a.detach()), intercept=float(b.detach()))


class SplitConformal:
    def __init__(self, alpha: float, width_budget: float, pe_threshold: int) -> None:
        self._alpha = alpha
        self._width_budget = width_budget
        self._pe_threshold = pe_threshold
        self._q_lo: QuantileLine | None = None
        self._q_hi: QuantileLine | None = None
        self._qhat: float = 0.0

    @property
    def qhat(self) -> float:
        return self._qhat

    def fit(
        self,
        score_train: FloatVec,
        y_train: FloatVec,
        score_cal: FloatVec,
        y_cal: FloatVec,
        steps: int,
        lr: float,
    ) -> None:
        lo = self._alpha / 2.0
        hi = 1.0 - self._alpha / 2.0
        self._q_lo = fit_quantile_line(score_train, y_train, lo, steps, lr)
        self._q_hi = fit_quantile_line(score_train, y_train, hi, steps, lr)
        cal_lo = self._q_lo(score_cal)
        cal_hi = self._q_hi(score_cal)
        scores = np.maximum(cal_lo - y_cal, y_cal - cal_hi)
        n = scores.shape[0]
        rank = int(np.ceil((n + 1) * (1.0 - self._alpha)))
        rank = min(max(rank, 1), n)
        self._qhat = float(np.sort(scores)[rank - 1])

    def _require(self) -> tuple[QuantileLine, QuantileLine]:
        if self._q_lo is None or self._q_hi is None:
            raise RuntimeError("conformal predictor used before fit")
        return self._q_lo, self._q_hi

    def predict(self, score: float, pe_count: int) -> PredictionSet:
        q_lo, q_hi = self._require()
        arr = np.array([score], dtype=np.float64)
        lower = float(q_lo(arr)[0]) - self._qhat
        upper = float(q_hi(arr)[0]) + self._qhat
        width = upper - lower
        refer = 1 if width > self._width_budget or pe_count < self._pe_threshold else 0
        return PredictionSet(lower=lower, upper=upper, refer=refer)

    def predict_many(self, scores: FloatVec, pe_counts: NDArray[np.int64]) -> list[PredictionSet]:
        return [self.predict(float(s), int(c)) for s, c in zip(scores, pe_counts, strict=True)]


class StratifiedConformal:
    def __init__(self, alpha: float, width_budget: float, pe_threshold: int) -> None:
        self._alpha = alpha
        self._width_budget = width_budget
        self._pe_threshold = pe_threshold
        self._strata: dict[str, SplitConformal] = {}

    def fit_stratum(
        self,
        key: str,
        score_train: FloatVec,
        y_train: FloatVec,
        score_cal: FloatVec,
        y_cal: FloatVec,
        steps: int,
        lr: float,
    ) -> None:
        model = SplitConformal(self._alpha, self._width_budget, self._pe_threshold)
        model.fit(score_train, y_train, score_cal, y_cal, steps, lr)
        self._strata[key] = model

    def predict(self, key: str, score: float, pe_count: int) -> PredictionSet:
        return self._strata[key].predict(score, pe_count)

    def strata(self) -> list[str]:
        return list(self._strata)
