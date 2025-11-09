from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from tiny_backprop.analysis.frontier_width import FrontierSummary, summarize_frontier
from tiny_backprop.analysis.hcp_criteria import HCPEvaluation, evaluate_hcp
from tiny_backprop.analysis.lower_bounds import lower_bound_report
from tiny_backprop.analysis.width_heuristics import OrderStats, evaluate_orders
from tiny_backprop.graph.ir import Graph, Node
from tiny_backprop.graph.topo import default_topological_order


@dataclass(frozen=True)
class OrderReport:
    strategy: str
    order: List[str]
    max_live: int


@dataclass(frozen=True)
class GraphAnalysisReport:
    primary_order: List[str]
    frontier: FrontierSummary
    hcp: HCPEvaluation
    lower_bounds: Dict[str, int]
    heuristic_orders: List[OrderReport]


def _coerce_order(graph: Graph, order: Sequence[Node] | Sequence[str]) -> List[Node]:
    if not order:
        return []
    first = order[0]
    if isinstance(first, Node):
        return list(order)  # type: ignore[return-value]
    return [graph.get_node(name) for name in order]  # type: ignore[arg-type]


def _orders_to_report(stats: Iterable[OrderStats]) -> List[OrderReport]:
    return [
        OrderReport(
            strategy=stat.strategy,
            order=[node.name for node in stat.order],
            max_live=stat.max_live,
        )
        for stat in stats
    ]


def analyze_graph(
    graph: Graph,
    *,
    order: Sequence[Node] | Sequence[str] | None = None,
    include_snapshots: bool = True,
) -> GraphAnalysisReport:
    heuristic_stats = evaluate_orders(graph)

    if order is None:
        if heuristic_stats:
            primary_nodes = heuristic_stats[0].order
        else:
            primary_nodes = default_topological_order(graph)
    else:
        primary_nodes = _coerce_order(graph, order)

    frontier_summary = summarize_frontier(graph, primary_nodes)
    if not include_snapshots:
        frontier_summary = FrontierSummary(
            snapshots=[],
            max_live=frontier_summary.max_live,
            argmax_step=frontier_summary.argmax_step,
            average_live=frontier_summary.average_live,
            final_live=frontier_summary.final_live,
        )

    return GraphAnalysisReport(
        primary_order=[node.name for node in primary_nodes],
        frontier=frontier_summary,
        hcp=evaluate_hcp(graph),
        lower_bounds=lower_bound_report(graph),
        heuristic_orders=_orders_to_report(heuristic_stats),
    )

