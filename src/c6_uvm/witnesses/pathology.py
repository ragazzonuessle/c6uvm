from __future__ import annotations

import numpy as np

from c6_uvm.intake.cohort import morphology_basis
from c6_uvm.record.types import FloatArray
from c6_uvm.statute.config import ModelConfig

_AXIS_TOKENS = ("BAP1", "H2A_K119ub", "chromatin_silencing", "PRC1")


class PathologyAgent:
    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        self._template = morphology_basis(cfg.tile_dim)

    def abstains(self, wsi: FloatArray | None) -> bool:
        return wsi is None

    def run(self, wsi: FloatArray | None, summary: str) -> float | None:
        if wsi is None:
            return None
        projection = float(np.dot(wsi, self._template))
        gate = 1.0 if any(tok in summary for tok in _AXIS_TOKENS) else 0.6
        return projection * gate
