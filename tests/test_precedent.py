from __future__ import annotations

from c6_uvm.precedent import build_um_graph, extract_subgraph, summarise
from c6_uvm.statute.config import DRIVERS, GraphConfig

GRAPH = build_um_graph()
CFG = GraphConfig()


def _drivers(*names: str) -> list[float]:
    return [1.0 if d in names else 0.0 for d in DRIVERS]


def test_extract_is_deterministic() -> None:
    a = extract_subgraph(GRAPH, _drivers("BAP1", "GNAQ"), 2, CFG)
    b = extract_subgraph(GRAPH, _drivers("BAP1", "GNAQ"), 2, CFG)
    assert a.nodes == b.nodes
    assert a.drugs == b.drugs


def test_seeds_are_only_mutated_genes() -> None:
    sub = extract_subgraph(GRAPH, _drivers("SF3B1"), 1, CFG)
    assert sub.seeds == ("SF3B1",)


def test_hop_budget_respected_for_class1() -> None:
    sub = extract_subgraph(GRAPH, _drivers("GNAQ"), 1, CFG)
    assert max(sub.distance.values()) <= CFG.hop_budget


def test_class2_expands_chromatin_axis() -> None:
    base = extract_subgraph(GRAPH, _drivers("BAP1"), 1, CFG)
    high = extract_subgraph(GRAPH, _drivers("BAP1"), 2, CFG)
    assert set(base.nodes).issubset(set(high.nodes))
    assert "chromatin_silencing" in high.nodes


def test_summary_mentions_drugs() -> None:
    sub = extract_subgraph(GRAPH, _drivers("BAP1", "GNAQ"), 2, CFG)
    text = summarise(GRAPH, sub)
    assert "drugs=" in text and "seeds=" in text


def test_string_threshold_filters_low_confidence() -> None:
    strict = extract_subgraph(GRAPH, _drivers("GNAQ"), 1, GraphConfig(string_tau=0.99))
    loose = extract_subgraph(GRAPH, _drivers("GNAQ"), 1, GraphConfig(string_tau=0.5))
    assert len(strict.edges) <= len(loose.edges)
