from __future__ import annotations

from attrs import frozen

from c6_uvm.precedent.graph import PathwayGraph
from c6_uvm.precedent.subgraph import Subgraph, extract_subgraph, summarise
from c6_uvm.record.types import FloatArray
from c6_uvm.statute.config import GraphConfig

_HIGH_RISK_AXIS = ("H2A_K119ub", "chromatin_silencing")


@frozen
class PathwayResult:
    subgraph: Subgraph
    summary: str
    drugs: tuple[str, ...]
    risk: float


class PathwayAgent:
    def __init__(self, graph: PathwayGraph, cfg: GraphConfig) -> None:
        self._graph = graph
        self._cfg = cfg

    def run(self, drivers: FloatArray, gep_class: int) -> PathwayResult:
        sub = extract_subgraph(self._graph, [float(v) for v in drivers], gep_class, self._cfg)
        summary = summarise(self._graph, sub)
        monosomy = float(drivers[5])
        axis_active = 1.0 if any(node in sub.nodes for node in _HIGH_RISK_AXIS) else 0.0
        risk = 0.5 * monosomy + 0.5 * axis_active
        return PathwayResult(subgraph=sub, summary=summary, drugs=sub.drugs, risk=risk)
