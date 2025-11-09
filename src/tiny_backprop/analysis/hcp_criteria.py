"""
Check whether a graph looks "height-compressible" under simple criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import StatisticsError, mean, quantiles
from typing import Dict, List, Sequence

from tiny_backprop.analysis.frontier_width import FrontierSummary, summarize_frontier
from tiny_backprop.graph.ir import Graph, Node


@dataclass
class HCPEvaluation:
    is_hcp_like: bool
    frontier_width: int
    average_frontier: float
    p95_frontier: int
    depth: int
    live_ratio: float
    max_fanout: int
    mean_fanout: float
    notes: List[str]


def _longest_path_depth(graph: Graph, order: Sequence[Node]) -> int:
    depth: Dict[str, int] = {}
    for node in order:
        if not node.inputs:
            depth[node.name] = 1
        else:
            depth[node.name] = 1 + max(depth[parent] for parent in node.inputs)
    return max(depth.values(), default=0)


def _fanout_stats(graph: Graph) -> tuple[int, float]:
    fanouts: List[int] = [len(graph.successors(name)) for name in graph.nodes]
    if not fanouts:
        return 0, 0.0
    return max(fanouts), float(mean(fanouts))


def evaluate_hcp(graph: Graph) -> HCPEvaluation:
    """
    Heuristically assess whether a graph appears height-compressible by examining
    frontier statistics, depth, and fan-out structure.
    """
    summary: FrontierSummary = summarize_frontier(graph)
    frontier_values = [snap.live_memory for snap in summary.snapshots]
    frontier_width = summary.max_live
    average_frontier = float(mean(frontier_values)) if frontier_values else 0.0
    if len(frontier_values) >= 5:
        try:
            p95 = int(round(quantiles(frontier_values, n=100, method="inclusive")[94]))
        except (StatisticsError, IndexError):
            p95 = frontier_width
    else:
        p95 = frontier_width

    order = [graph.get_node(snap.node) for snap in summary.snapshots]
    depth = _longest_path_depth(graph, order)
    total_activation = graph.total_activation_size()
    live_ratio = (frontier_width / total_activation) if total_activation else 0.0
    max_fanout, mean_fanout = _fanout_stats(graph)

    notes: List[str] = []

    if frontier_width == 0:
        notes.append("Graph has zero live frontier; trivially compressible.")
        is_hcp = True
    else:
        is_hcp = live_ratio <= 0.35 and p95 <= frontier_width * 1.05 and max_fanout <= 16

        if live_ratio <= 0.35:
            notes.append(f"Peak frontier is {live_ratio:.1%} of total activation.")
        else:
            notes.append(f"Peak frontier is {live_ratio:.1%} of total activation (high).")

        if p95 < frontier_width * 0.9:
            notes.append("Frontier tail is substantially lower than peak (good locality).")
        elif len(frontier_values) > 0:
            notes.append("Frontier tail remains near peak; expect limited savings.")

        if max_fanout <= 8:
            notes.append(f"Max fan-out {max_fanout}; dependencies are localized.")
        else:
            notes.append(f"Max fan-out {max_fanout}; wide branching may limit compression.")

    if depth <= len(graph.nodes) ** 0.5:
        notes.append(f"Depth {depth} relative to node count suggests shallow graph.")
    else:
        notes.append(f"Depth {depth} indicates long sequential structure.")

    return HCPEvaluation(
        is_hcp_like=is_hcp,
        frontier_width=frontier_width,
        average_frontier=average_frontier,
        p95_frontier=p95,
        depth=depth,
        live_ratio=live_ratio,
        max_fanout=max_fanout,
        mean_fanout=mean_fanout,
        notes=notes,
    )
