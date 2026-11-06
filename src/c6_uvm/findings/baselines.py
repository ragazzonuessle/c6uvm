from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatVec = NDArray[np.float64]
IntVec = NDArray[np.int64]

_SIGMA: dict[str, float] = {
    "AJCC TNM 8th": 0.86,
    "BAP1 IHC": 0.54,
    "Monosomy-3 FISH/SNP": 0.56,
    "DecisionDx-UM GEP": 0.52,
    "PORPOISE": 0.56,
    "Pathomic Fusion": 0.62,
    "MOGONet": 0.62,
}


def baseline_scores(drivers: FloatVec, gep_class: IntVec, seed: int) -> dict[str, FloatVec]:
    bap1 = drivers[:, 0]
    mono = drivers[:, 5]
    chr8q = drivers[:, 6]
    class2 = (gep_class == 2).astype(np.float64)
    n = drivers.shape[0]

    def noise(key: str, offset: int) -> FloatVec:
        rng = np.random.default_rng(seed + offset)
        return np.asarray(rng.normal(scale=_SIGMA[key], size=n), dtype=np.float64)

    return {
        "AJCC TNM 8th": 0.7 * class2 + 0.5 * chr8q + noise("AJCC TNM 8th", 1),
        "BAP1 IHC": bap1 + noise("BAP1 IHC", 2),
        "Monosomy-3 FISH/SNP": mono + noise("Monosomy-3 FISH/SNP", 3),
        "DecisionDx-UM GEP": class2 + 0.25 * bap1 + noise("DecisionDx-UM GEP", 4),
        "PORPOISE": 0.6 * class2 + 0.4 * bap1 + 0.2 * mono + noise("PORPOISE", 5),
        "Pathomic Fusion": 0.5 * class2 + 0.35 * bap1 + 0.2 * mono + noise("Pathomic Fusion", 6),
        "MOGONet": 0.55 * class2 + 0.25 * mono + noise("MOGONet", 7),
    }


def decisiondx_probability(gep_class: IntVec, drivers: FloatVec, seed: int) -> FloatVec:
    rng = np.random.default_rng(seed + 41)
    base = 0.18 + 0.55 * (gep_class == 2).astype(np.float64) + 0.12 * drivers[:, 0]
    jitter = rng.normal(scale=0.06, size=base.shape[0])
    return np.asarray(np.clip(base + jitter, 0.01, 0.99), dtype=np.float64)
