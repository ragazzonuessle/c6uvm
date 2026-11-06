from __future__ import annotations

from attrs import frozen

from c6_uvm.bench.order import canonical_order, validate_schedule
from c6_uvm.bench.system import C6System
from c6_uvm.record.store import EvidenceStore
from c6_uvm.record.types import AgentEvidence, Patient
from c6_uvm.ruling.conformal import SplitConformal


@frozen
class CaseOutcome:
    pid: str
    risk: float
    gep_class: int
    subgraph_nodes: tuple[str, ...]
    drugs: tuple[str, ...]
    lower: float
    upper: float
    refer: int
    trace: tuple[tuple[str, str, float], ...]


class Orchestrator:
    def __init__(self, system: C6System, conformal: SplitConformal) -> None:
        self._system = system
        self._conformal = conformal

    @property
    def system(self) -> C6System:
        return self._system

    def run(self, patient: Patient) -> CaseOutcome:
        schedule = list(canonical_order())
        if not validate_schedule(schedule):
            raise RuntimeError("KG-first invariant violated")
        store = EvidenceStore()
        table = self._system.score([patient])
        store.append(
            AgentEvidence(
                source="subtype",
                node_id="GEP",
                score=float(table.subtype[0]),
                payload={"class": float(table.class_hat[0])},
                provenance=("DecisionDx-UM",),
            )
        )
        for node in table.summaries[0].split("|"):
            store.append(
                AgentEvidence(
                    source="pathway",
                    node_id=node[:24],
                    score=float(table.pathway[0]),
                    payload={"pe": float(table.pe_counts[0])},
                    provenance=("Reactome", "OmniPath", "STRING", "KEGG"),
                )
            )
        store.append(
            AgentEvidence(
                source="pathology",
                node_id="morphology",
                score=float(table.pathology[0]),
                payload={},
                provenance=("CONCH",),
            )
        )
        for drug in table.drugs[0]:
            store.append(
                AgentEvidence(
                    source="therapy",
                    node_id=drug,
                    score=float(table.therapy[0]),
                    payload={},
                    provenance=("OncoKB", "DrugBank"),
                )
            )
        risk = float(self._system.risk_from_table(table)[0])
        store.append(
            AgentEvidence(
                source="router",
                node_id="risk",
                score=risk,
                payload={},
                provenance=(self._system.router.backbone,),
            )
        )
        prediction = self._conformal.predict(risk, int(table.pe_counts[0]))
        store.append(
            AgentEvidence(
                source="conformal",
                node_id="prediction",
                score=prediction.width,
                payload={"refer": float(prediction.refer)},
                provenance=("split-conformal",),
            )
        )
        return CaseOutcome(
            pid=patient.pid,
            risk=risk,
            gep_class=int(table.class_hat[0]),
            subgraph_nodes=tuple(table.summaries[0].split("|")),
            drugs=table.drugs[0],
            lower=prediction.lower,
            upper=prediction.upper,
            refer=prediction.refer,
            trace=tuple(store.trace()),
        )
