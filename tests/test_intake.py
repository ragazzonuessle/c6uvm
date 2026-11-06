from __future__ import annotations

import numpy as np

from c6_uvm.intake.cohort import build_dataset, driver_frequencies
from c6_uvm.statute.config import DRIVERS, CohortConfig


def test_cohort_sizes_match_config() -> None:
    cfg = CohortConfig()
    data = build_dataset(cfg)
    for name, size in cfg.sizes.items():
        assert len(data[name]) == size


def test_generation_is_reproducible() -> None:
    cfg = CohortConfig()
    a = build_dataset(cfg)["TCGA-UVM"]
    b = build_dataset(cfg)["TCGA-UVM"]
    assert np.allclose(a.times(), b.times())
    assert a.patients[0].pid == b.patients[0].pid


def test_bap1_frequency_in_plausible_band() -> None:
    cfg = CohortConfig()
    pooled = [p for c in build_dataset(cfg).values() for p in c.patients]
    bap1_index = DRIVERS.index("BAP1")
    freq = float(np.mean([p.drivers[bap1_index] for p in pooled]))
    assert 0.35 <= freq <= 0.60


def test_event_and_wsi_presence() -> None:
    cfg = CohortConfig()
    cohort = build_dataset(cfg)["TCGA-UVM"]
    events = cohort.events()
    assert events.sum() > 0
    freqs = driver_frequencies(cohort)
    assert freqs.shape == (len(DRIVERS),)
    assert any(p.has_wsi for p in cohort.patients)
