from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import frozen
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

FloatArray = NDArray[np.float64]


@frozen(kw_only=True)
class Patient:
    pid: str
    cohort: str
    rna: FloatArray
    drivers: FloatArray
    wsi: FloatArray | None
    gep_class: int
    time: float
    event: int
    latent: float

    @property
    def has_wsi(self) -> bool:
        return self.wsi is not None


@frozen(kw_only=True)
class Cohort:
    name: str
    patients: tuple[Patient, ...]

    def __len__(self) -> int:
        return len(self.patients)

    def times(self) -> FloatArray:
        return np.array([p.time for p in self.patients], dtype=np.float64)

    def events(self) -> NDArray[np.int64]:
        return np.array([p.event for p in self.patients], dtype=np.int64)

    def latents(self) -> FloatArray:
        return np.array([p.latent for p in self.patients], dtype=np.float64)


@frozen(kw_only=True)
class TherapyCandidate:
    name: str
    target: str
    oncokb_tier: int
    median_log_ic50: float
    path_length: int
    provenance: tuple[str, ...]


@frozen(kw_only=True)
class AgentEvidence:
    source: str
    node_id: str
    score: float
    payload: dict[str, float]
    provenance: tuple[str, ...]


def stack_rna(patients: Sequence[Patient]) -> FloatArray:
    return np.stack([p.rna for p in patients]).astype(np.float64)
