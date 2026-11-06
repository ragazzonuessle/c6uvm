from __future__ import annotations

import numpy as np

from c6_uvm.bench import Orchestrator, canonical_order, validate_schedule
from c6_uvm.bench.system import C6System
from c6_uvm.intake.cohort import build_dataset
from c6_uvm.proceedings.split import pooled_split
from c6_uvm.record.seeding import set_seed
from c6_uvm.ruling.conformal import SplitConformal
from c6_uvm.statute.config import CohortConfig, ExperimentConfig


def test_canonical_order_is_valid() -> None:
    assert validate_schedule(list(canonical_order())) is True


def test_out_of_order_schedule_rejected() -> None:
    bad = ["conformal", "router", "pathway", "subtype", "pathology", "therapy"]
    assert validate_schedule(bad) is False


def _small_cfg() -> ExperimentConfig:
    sizes = {"TCGA-UVM": 24, "GSE22138": 20, "GSE44295": 18, "GSE84976": 12}
    return ExperimentConfig(cohort=CohortConfig(sizes=sizes))


def test_orchestrator_produces_auditable_trace() -> None:
    cfg = _small_cfg()
    set_seed(0)
    data = build_dataset(cfg.cohort)
    patients = [p for c in data.values() for p in c.patients]
    system = C6System(cfg)
    system.fit(patients)
    split = pooled_split(patients, 0.386, 1)
    risk_tr = system.c6_risk(list(split.train))
    risk_cal = system.c6_risk(list(split.cal))
    y_tr = np.log(np.array([p.time for p in split.train]) + 1e-6)
    y_cal = np.log(np.array([p.time for p in split.cal]) + 1e-6)
    conformal = SplitConformal(0.1, 6.6, 4)
    conformal.fit(risk_tr, y_tr, risk_cal, y_cal, 40, 0.05)
    outcome = Orchestrator(system, conformal).run(patients[0])
    sources = [t[0] for t in outcome.trace]
    assert sources.index("subtype") < sources.index("router")
    assert sources.index("router") < sources.index("conformal")
    assert np.isfinite(outcome.risk)
    assert outcome.upper >= outcome.lower
