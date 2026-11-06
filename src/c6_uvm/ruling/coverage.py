from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from attrs import frozen
from numpy.typing import NDArray

from c6_uvm.ruling.conformal import PredictionSet


def marginal_band(n: int, alpha: float) -> tuple[float, float]:
    lower = 1.0 - alpha
    upper = 1.0 - alpha + 1.0 / (n + 1)
    return lower, upper


def empirical_coverage(sets: Sequence[PredictionSet], y: NDArray[np.float64]) -> float:
    hits = sum(1 for ps, target in zip(sets, y, strict=True) if ps.contains(float(target)))
    return hits / len(y) if len(y) else 0.0


@frozen
class HallucinationBound:
    rho: float
    eps_vlm: float
    eta: float
    rho_plus: float
    holds: bool


def hallucination_bound(rho: float, eps_vlm: float, eta: float) -> HallucinationBound:
    rho_plus = rho + eps_vlm * (1.0 - rho)
    return HallucinationBound(
        rho=rho,
        eps_vlm=eps_vlm,
        eta=eta,
        rho_plus=rho_plus,
        holds=rho_plus <= eta,
    )
