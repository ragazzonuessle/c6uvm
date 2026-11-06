from __future__ import annotations

import numpy as np
from attrs import frozen
from numpy.typing import NDArray

from c6_uvm.record.types import Patient


@frozen
class Split:
    train: tuple[Patient, ...]
    cal: tuple[Patient, ...]
    test: tuple[Patient, ...]


def pooled_split(patients: list[Patient], cal_fraction: float, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(patients))
    n_cal = max(2, round(cal_fraction * len(patients)))
    cal_idx = idx[:n_cal]
    rest = idx[n_cal:]
    cut = max(2, round(0.3 * len(rest)))
    train_idx = rest[:cut]
    test_idx = rest[cut:]

    def pick(group: NDArray[np.int64]) -> tuple[Patient, ...]:
        return tuple(patients[int(i)] for i in group)

    return Split(train=pick(train_idx), cal=pick(cal_idx), test=pick(test_idx))
