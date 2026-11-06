from __future__ import annotations

from pathlib import Path

from c6_uvm.proceedings.pipeline import run_main
from c6_uvm.statute.loader import load_experiment

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_run_main_produces_four_rows() -> None:
    cfg = load_experiment(ROOT / "configs" / "experiment" / "_smoke.yaml")
    report = run_main(cfg)
    assert len(report.rows) == 4
    assert 0.0 <= report.coverage <= 1.0
    assert report.pooled_lo <= report.pooled_c6 <= report.pooled_hi
    for row in report.rows:
        assert 0.4 <= row.c6 <= 1.0


def test_smoke_subtype_loss_decreases() -> None:
    import numpy as np

    from c6_uvm.intake.cohort import build_dataset
    from c6_uvm.record.types import stack_rna
    from c6_uvm.witnesses import SubtypeAgent

    cfg = load_experiment(ROOT / "configs" / "experiment" / "_smoke.yaml")
    cohort = build_dataset(cfg.cohort)["TCGA-UVM"]
    rna = stack_rna(cohort.patients)
    classes = np.array([p.gep_class for p in cohort.patients], dtype=np.int64)
    agent = SubtypeAgent(cfg.model)
    history = agent.fit(rna, classes, cfg.train)
    assert history[-1] < history[0]
