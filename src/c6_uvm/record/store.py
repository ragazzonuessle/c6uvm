from __future__ import annotations

from collections import OrderedDict

from c6_uvm.record.types import AgentEvidence


class EvidenceStore:
    def __init__(self) -> None:
        self._by_node: OrderedDict[str, list[AgentEvidence]] = OrderedDict()
        self._order: list[AgentEvidence] = []

    def append(self, evidence: AgentEvidence) -> None:
        self._by_node.setdefault(evidence.node_id, []).append(evidence)
        self._order.append(evidence)

    def by_node(self, node_id: str) -> list[AgentEvidence]:
        return list(self._by_node.get(node_id, []))

    def sources(self) -> list[str]:
        seen: list[str] = []
        for ev in self._order:
            if ev.source not in seen:
                seen.append(ev.source)
        return seen

    def trace(self) -> list[tuple[str, str, float]]:
        return [(ev.source, ev.node_id, ev.score) for ev in self._order]

    def __len__(self) -> int:
        return len(self._order)
