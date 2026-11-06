from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from attrs import frozen
from numpy.typing import NDArray
from torch import nn

from c6_uvm.bench.router import Router
from c6_uvm.precedent.graph import build_um_graph
from c6_uvm.record.types import Patient, stack_rna
from c6_uvm.statute.config import ExperimentConfig
from c6_uvm.witnesses import PathologyAgent, PathwayAgent, SubtypeAgent, TherapyAgent

FloatVec = NDArray[np.float64]
IntVec = NDArray[np.int64]

_SCORING_AGENTS = ("subtype", "pathway", "pathology", "therapy")


@frozen
class ScoreTable:
    subtype: FloatVec
    pathway: FloatVec
    pathology: FloatVec
    therapy: FloatVec
    pe_counts: IntVec
    class_hat: IntVec
    summaries: tuple[str, ...]
    drugs: tuple[tuple[str, ...], ...]

    def scoring_view(self, agents: Sequence[str]) -> dict[str, FloatVec]:
        view: dict[str, FloatVec] = {}
        for name in _SCORING_AGENTS:
            if name in agents:
                view[name] = getattr(self, name)
        return view


class C6System:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self._cfg = cfg
        graph = build_um_graph()
        self._subtype = SubtypeAgent(cfg.model)
        self._pathway = PathwayAgent(graph, cfg.graph)
        self._pathology = PathologyAgent(cfg.model)
        self._therapy = TherapyAgent(graph)
        self._router = Router(cfg.model.router_weights, cfg.router_backbone)
        self._has_subtype = "subtype" in cfg.agents
        self._kg_first = cfg.kg_first
        self._path_fallback = 0.0
        self._fitted = False

    @property
    def router(self) -> Router:
        return self._router

    @property
    def subtype_head(self) -> nn.Module:
        return self._subtype.head

    def fit(self, train: Sequence[Patient]) -> list[float]:
        rna = stack_rna(train)
        classes = np.array([p.gep_class for p in train], dtype=np.int64)
        history = self._subtype.fit(rna, classes, self._cfg.train)
        observed: list[float] = []
        for patient in train:
            class_hat = self._class_hat(patient)
            pw = self._pathway.run(patient.drivers, class_hat)
            summary = pw.summary if self._kg_first else ""
            value = self._pathology.run(patient.wsi, summary)
            if value is not None:
                observed.append(value)
        self._path_fallback = float(np.mean(observed)) if observed else 0.0
        table = self.score(train)
        self._router.fit(table.scoring_view(self._cfg.agents))
        self._fitted = True
        return history

    def _class_hat(self, patient: Patient) -> int:
        if self._has_subtype:
            return self._subtype.emit_class(patient.rna)
        return 1

    def score(self, patients: Sequence[Patient]) -> ScoreTable:
        n = len(patients)
        subtype = np.zeros(n, dtype=np.float64)
        pathway = np.zeros(n, dtype=np.float64)
        pathology = np.zeros(n, dtype=np.float64)
        therapy = np.zeros(n, dtype=np.float64)
        pe_counts = np.zeros(n, dtype=np.int64)
        class_hat = np.zeros(n, dtype=np.int64)
        summaries: list[str] = []
        drugs: list[tuple[str, ...]] = []
        for i, patient in enumerate(patients):
            chat = self._class_hat(patient)
            class_hat[i] = chat
            subtype[i] = self._subtype.prob_class2(patient.rna)
            pw = self._pathway.run(patient.drivers, chat)
            pathway[i] = pw.risk
            pe_counts[i] = len(pw.subgraph.nodes)
            summary = pw.summary if self._kg_first else ""
            value = self._pathology.run(patient.wsi, summary)
            pathology[i] = self._path_fallback if value is None else value
            therapy[i] = self._therapy.run(pw.subgraph).score
            summaries.append(pw.summary)
            drugs.append(pw.drugs)
        return ScoreTable(
            subtype=subtype,
            pathway=pathway,
            pathology=pathology,
            therapy=therapy,
            pe_counts=pe_counts,
            class_hat=class_hat,
            summaries=tuple(summaries),
            drugs=tuple(drugs),
        )

    def c6_risk(self, patients: Sequence[Patient]) -> FloatVec:
        if not self._fitted:
            raise RuntimeError("system used before fit")
        table = self.score(patients)
        return self._router.aggregate(table.scoring_view(self._cfg.agents))

    def risk_from_table(self, table: ScoreTable) -> FloatVec:
        return self._router.aggregate(table.scoring_view(self._cfg.agents))
