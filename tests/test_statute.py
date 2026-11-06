from __future__ import annotations

from pathlib import Path

from c6_uvm.statute import dump_config, load_experiment, structure_config
from c6_uvm.statute.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]


def test_main_experiment_loads_paper_defaults() -> None:
    cfg = load_experiment(ROOT / "configs" / "experiment" / "main.yaml")
    assert cfg.name == "main"
    assert cfg.conformal.alpha == 0.10
    assert cfg.graph.hop_budget == 3
    assert cfg.graph.string_tau == 0.7
    assert cfg.eval.bootstrap == 1000
    assert cfg.cohort.sizes["TCGA-UVM"] == 80


def test_overrides_apply() -> None:
    cfg = load_experiment(
        ROOT / "configs" / "experiment" / "main.yaml",
        ["conformal.alpha=0.2", "router_backbone=gpt-4o"],
    )
    assert cfg.conformal.alpha == 0.2
    assert cfg.router_backbone == "gpt-4o"


def test_roundtrip_structure() -> None:
    cfg = ExperimentConfig()
    again = structure_config(dump_config(cfg))
    assert again.name == cfg.name
    assert again.cohort.risk_scale == cfg.cohort.risk_scale
