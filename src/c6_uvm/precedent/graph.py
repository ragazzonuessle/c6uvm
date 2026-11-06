from __future__ import annotations

from attrs import frozen

GENE = "gene"
PATHWAY = "pathway"
DRUG = "drug"


@frozen(kw_only=True)
class Edge:
    src: str
    dst: str
    relation: str
    confidence: float
    provenance: str


@frozen
class DrugMeta:
    name: str
    oncokb_tier: int
    median_log_ic50: float


class PathwayGraph:
    def __init__(
        self,
        nodes: dict[str, str],
        edges: list[Edge],
        drug_meta: dict[str, DrugMeta],
    ) -> None:
        self._nodes = dict(nodes)
        self._edges = list(edges)
        self._drug_meta = dict(drug_meta)
        self._adj: dict[str, list[Edge]] = {n: [] for n in nodes}
        for edge in edges:
            self._adj[edge.src].append(edge)

    @property
    def nodes(self) -> dict[str, str]:
        return dict(self._nodes)

    @property
    def edges(self) -> list[Edge]:
        return list(self._edges)

    def kind(self, node: str) -> str:
        return self._nodes[node]

    def drug_nodes(self) -> set[str]:
        return {n for n, k in self._nodes.items() if k == DRUG}

    def drug_meta(self, node: str) -> DrugMeta:
        return self._drug_meta[node]

    def neighbours(self, node: str) -> list[Edge]:
        return list(self._adj.get(node, []))


def build_um_graph() -> PathwayGraph:
    nodes: dict[str, str] = {
        "GNAQ": GENE,
        "GNA11": GENE,
        "BAP1": GENE,
        "SF3B1": GENE,
        "EIF1AX": GENE,
        "monosomy3": GENE,
        "chr8q": GENE,
        "PLCB4": PATHWAY,
        "PKC": PATHWAY,
        "MAPK": PATHWAY,
        "YAP_TAZ": PATHWAY,
        "PRC1": PATHWAY,
        "PRC2": PATHWAY,
        "EZH2": PATHWAY,
        "H2A_K119ub": PATHWAY,
        "H3K27me3": PATHWAY,
        "chromatin_silencing": PATHWAY,
        "spliceosome": PATHWAY,
        "U2AF": PATHWAY,
        "aberrant_splicing": PATHWAY,
        "initiation_complex": PATHWAY,
        "HDAC": DRUG,
        "PARP": DRUG,
        "GPX4": DRUG,
        "PKC_target": DRUG,
        "MET": DRUG,
        "gp100": DRUG,
    }
    edges: list[Edge] = [
        Edge(src="GNAQ", dst="PLCB4", relation="activates", confidence=0.92, provenance="Reactome"),
        Edge(src="GNAQ", dst="PKC", relation="activates", confidence=0.90, provenance="Reactome"),
        Edge(
            src="GNAQ", dst="YAP_TAZ", relation="activates", confidence=0.88, provenance="OmniPath"
        ),
        Edge(src="GNA11", dst="PKC", relation="activates", confidence=0.90, provenance="Reactome"),
        Edge(
            src="GNA11", dst="YAP_TAZ", relation="activates", confidence=0.87, provenance="OmniPath"
        ),
        Edge(src="PKC", dst="MAPK", relation="activates", confidence=0.91, provenance="STRING"),
        Edge(
            src="PKC",
            dst="PKC_target",
            relation="drug-target",
            confidence=0.80,
            provenance="OncoKB",
        ),
        Edge(src="MAPK", dst="MET", relation="drug-target", confidence=0.72, provenance="OncoKB"),
        Edge(src="BAP1", dst="PRC1", relation="inhibits", confidence=0.89, provenance="Reactome"),
        Edge(
            src="BAP1",
            dst="PARP",
            relation="synthetic-lethal",
            confidence=0.74,
            provenance="OmniPath",
        ),
        Edge(
            src="PRC1",
            dst="H2A_K119ub",
            relation="activates",
            confidence=0.90,
            provenance="Reactome",
        ),
        Edge(
            src="PRC2", dst="H3K27me3", relation="activates", confidence=0.88, provenance="Reactome"
        ),
        Edge(src="PRC2", dst="EZH2", relation="binds", confidence=0.85, provenance="STRING"),
        Edge(
            src="H2A_K119ub",
            dst="chromatin_silencing",
            relation="activates",
            confidence=0.86,
            provenance="Reactome",
        ),
        Edge(
            src="H3K27me3",
            dst="chromatin_silencing",
            relation="activates",
            confidence=0.84,
            provenance="Reactome",
        ),
        Edge(
            src="chromatin_silencing",
            dst="HDAC",
            relation="drug-target",
            confidence=0.78,
            provenance="OncoKB",
        ),
        Edge(
            src="H2A_K119ub",
            dst="HDAC",
            relation="drug-target",
            confidence=0.76,
            provenance="DrugBank",
        ),
        Edge(
            src="chromatin_silencing",
            dst="GPX4",
            relation="drug-target",
            confidence=0.71,
            provenance="DrugBank",
        ),
        Edge(
            src="SF3B1",
            dst="spliceosome",
            relation="activates",
            confidence=0.90,
            provenance="Reactome",
        ),
        Edge(src="spliceosome", dst="U2AF", relation="binds", confidence=0.83, provenance="STRING"),
        Edge(
            src="spliceosome",
            dst="aberrant_splicing",
            relation="activates",
            confidence=0.80,
            provenance="OmniPath",
        ),
        Edge(
            src="EIF1AX",
            dst="initiation_complex",
            relation="activates",
            confidence=0.85,
            provenance="Reactome",
        ),
        Edge(src="GNAQ", dst="gp100", relation="drug-target", confidence=0.70, provenance="OncoKB"),
        Edge(
            src="GNA11", dst="gp100", relation="drug-target", confidence=0.70, provenance="OncoKB"
        ),
    ]
    drug_meta: dict[str, DrugMeta] = {
        "HDAC": DrugMeta("vorinostat", 3, -1.20),
        "PARP": DrugMeta("niraparib", 3, -0.65),
        "GPX4": DrugMeta("RSL3-ferroptosis", 4, 0.10),
        "PKC_target": DrugMeta("darovasertib", 2, -1.55),
        "MET": DrugMeta("crizotinib", 3, -0.90),
        "gp100": DrugMeta("tebentafusp", 1, -1.80),
    }
    return PathwayGraph(nodes=nodes, edges=edges, drug_meta=drug_meta)
