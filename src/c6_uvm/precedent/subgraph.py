from __future__ import annotations

from collections import deque

from attrs import frozen

from c6_uvm.precedent.graph import Edge, PathwayGraph
from c6_uvm.statute.config import DRIVERS, GraphConfig

_CLASS2_EXPANSION = ("BAP1", "PRC1", "PRC2", "H2A_K119ub")


@frozen
class Subgraph:
    seeds: tuple[str, ...]
    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]
    distance: dict[str, int]
    drugs: tuple[str, ...]


def _seed_genes(drivers: list[float], graph: PathwayGraph) -> list[str]:
    present = graph.nodes
    return [DRIVERS[i] for i, flag in enumerate(drivers) if flag > 0.5 and DRIVERS[i] in present]


def _bfs(
    graph: PathwayGraph,
    frontier: list[str],
    distance: dict[str, int],
    hop_budget: int,
    tau: float,
    kept_edges: list[Edge],
    visited_pairs: set[tuple[str, str, str]],
) -> None:
    queue: deque[str] = deque(frontier)
    while queue:
        node = queue.popleft()
        depth = distance[node]
        if depth >= hop_budget:
            continue
        for edge in graph.neighbours(node):
            if edge.confidence < tau:
                continue
            key = (edge.src, edge.dst, edge.relation)
            if key not in visited_pairs:
                visited_pairs.add(key)
                kept_edges.append(edge)
            if edge.dst not in distance:
                distance[edge.dst] = depth + 1
                queue.append(edge.dst)


def extract_subgraph(
    graph: PathwayGraph,
    drivers: list[float],
    gep_class: int,
    cfg: GraphConfig,
) -> Subgraph:
    seeds = _seed_genes(drivers, graph)
    distance: dict[str, int] = {s: 0 for s in seeds}
    kept_edges: list[Edge] = []
    visited_pairs: set[tuple[str, str, str]] = set()
    _bfs(graph, seeds, distance, cfg.hop_budget, cfg.string_tau, kept_edges, visited_pairs)

    if gep_class == 2 and cfg.class2_expand_hops > 0:
        present = graph.nodes
        anchors = [n for n in _CLASS2_EXPANSION if n in present]
        for anchor in anchors:
            distance.setdefault(anchor, cfg.hop_budget)
        _bfs(
            graph,
            anchors,
            distance,
            cfg.hop_budget + cfg.class2_expand_hops,
            cfg.string_tau,
            kept_edges,
            visited_pairs,
        )

    drug_nodes = graph.drug_nodes()
    reached_drugs = [n for n in distance if n in drug_nodes]
    reached_drugs.sort(key=lambda n: (distance[n], graph.drug_meta(n).median_log_ic50))
    nodes = tuple(sorted(distance))
    return Subgraph(
        seeds=tuple(seeds),
        nodes=nodes,
        edges=tuple(kept_edges),
        distance=dict(distance),
        drugs=tuple(reached_drugs),
    )


def summarise(graph: PathwayGraph, sub: Subgraph) -> str:
    parts: list[str] = []
    parts.append("seeds=" + ",".join(sub.seeds))
    parts.append("nodes=" + ",".join(sub.nodes))
    rels = sorted({f"{e.src}-{e.relation}->{e.dst}" for e in sub.edges})
    parts.append("edges=" + ";".join(rels))
    drug_summary = [f"{graph.drug_meta(d).name}@{sub.distance[d]}" for d in sub.drugs]
    parts.append("drugs=" + ",".join(drug_summary))
    return "|".join(parts)
