from __future__ import annotations

from attrs import frozen

from c6_uvm.precedent.graph import PathwayGraph
from c6_uvm.precedent.subgraph import Subgraph
from c6_uvm.record.types import TherapyCandidate

_TIER_LABEL = {1: "FDA-approved", 2: "clinical-trial", 3: "investigational", 4: "emerging"}


@frozen
class TherapyResult:
    candidates: tuple[TherapyCandidate, ...]
    score: float


class TherapyAgent:
    def __init__(self, graph: PathwayGraph) -> None:
        self._graph = graph

    def run(self, sub: Subgraph) -> TherapyResult:
        candidates: list[TherapyCandidate] = []
        for node in sub.drugs:
            meta = self._graph.drug_meta(node)
            provenance = tuple(
                f"{e.src}-{e.relation}->{e.dst}@{e.provenance}" for e in sub.edges if e.dst == node
            )
            candidates.append(
                TherapyCandidate(
                    name=meta.name,
                    target=node,
                    oncokb_tier=meta.oncokb_tier,
                    median_log_ic50=meta.median_log_ic50,
                    path_length=sub.distance[node],
                    provenance=provenance or (f"{node}@curated",),
                )
            )
        if candidates:
            score = float(sum((4 - c.oncokb_tier) for c in candidates)) / (4.0 * len(candidates))
        else:
            score = 0.0
        return TherapyResult(candidates=tuple(candidates), score=score)

    @staticmethod
    def tier_label(tier: int) -> str:
        return _TIER_LABEL.get(tier, "unknown")
