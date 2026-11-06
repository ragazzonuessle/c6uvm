from __future__ import annotations

import numpy as np

from c6_uvm.statute.config import ModelConfig, TrainConfig
from c6_uvm.witnesses import PathologyAgent, SubtypeAgent


def test_subtype_overfits_single_batch() -> None:
    rng = np.random.default_rng(0)
    cfg = ModelConfig(rna_dim=15, subtype_hidden=32)
    agent = SubtypeAgent(cfg)
    direction = rng.normal(size=15)
    labels = np.array([1, 2] * 16, dtype=np.int64)
    rna = np.stack([(2.5 if c == 2 else -2.5) * direction for c in labels]).astype(np.float64)
    history = agent.fit(rna, labels, TrainConfig(subtype_steps=300, subtype_lr=0.02))
    assert history[-1] < history[0]
    preds = (agent.predict_proba(rna) >= 0.5).astype(np.int64) + 1
    assert float(np.mean(preds == labels)) >= 0.9


def test_pathology_abstains_without_wsi() -> None:
    agent = PathologyAgent(ModelConfig())
    assert agent.abstains(None) is True
    assert agent.run(None, "seeds=BAP1") is None


def test_pathology_reads_epithelioid_signal() -> None:
    from c6_uvm.intake.cohort import morphology_basis

    cfg = ModelConfig()
    agent = PathologyAgent(cfg)
    template = morphology_basis(cfg.tile_dim)
    high = agent.run(2.0 * template, "seeds=BAP1|nodes=H2A_K119ub")
    low = agent.run(-2.0 * template, "seeds=BAP1|nodes=H2A_K119ub")
    assert high is not None and low is not None
    assert high > low
