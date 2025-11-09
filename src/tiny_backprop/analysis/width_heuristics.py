"""
Heuristics to improve topological order to reduce frontier width.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from tiny_backprop.analysis.frontier_width import compute_frontier_width
from tiny_backprop.graph.ir import Graph, Node
from tiny_backprop.graph.topo import default_topological_order


ScoreFn = Callable[[Graph, Sequence[str]], int]


@dataclass(frozen=True)
class OrderStats:
    strategy: str
    order: List[Node]
    max_live: int


def _build_graph_metadata(graph: Graph) -> Tuple[Dict[str, int], Dict[str, List[str]], Dict[Tuple[str, str], int]]:
    indegree: Dict[str, int] = {name: 0 for name in graph.nodes}
    successors: Dict[str, List[str]] = {name: [] for name in graph.nodes}
    edge_counts: Dict[Tuple[str, str], int] = {}

    for node in graph.nodes.values():
        for parent in node.inputs:
            indegree[node.name] += 1
            successors.setdefault(parent, []).append(node.name)
            edge_counts[(parent, node.name)] = edge_counts.get((parent, node.name), 0) + 1

    return indegree, successors, edge_counts


def greedy_min_cutwidth_order(graph: Graph) -> List[Node]:
    """
    Topological order that greedily minimises the instantaneous live activation size.
    """
    indegree, successors, edge_counts = _build_graph_metadata(graph)

    ready: List[str] = sorted([name for name, deg in indegree.items() if deg == 0])
    if not ready:
        return []

    remaining_uses: Dict[str, int] = {
        name: sum(edge_counts[(name, succ)] for succ in successors.get(name, []))
        for name in graph.nodes
    }

    live: Dict[str, int] = {}
    live_memory = 0
    order: List[str] = []

    def forecast(node_name: str) -> Tuple[int, int, str]:
        size_node = graph.get_node(node_name).outputs_size
        predicted = live_memory + size_node

        for parent in graph.get_node(node_name).inputs:
            usage = edge_counts.get((parent, node_name), 1)
            remaining = remaining_uses[parent] - usage
            if remaining <= 0 and parent in live:
                predicted -= live[parent]
        # tie-break on node size (prefer smaller) then name
        return (predicted, size_node, node_name)

    while ready:
        ready.sort(key=forecast)
        current = ready.pop(0)

        node = graph.get_node(current)
        order.append(current)

        if node.outputs_size > 0:
            live[current] = node.outputs_size
            live_memory += node.outputs_size

        for parent in node.inputs:
            usage = edge_counts.get((parent, current), 1)
            remaining_uses[parent] -= usage
            if remaining_uses[parent] <= 0 and parent in live:
                live_memory -= live.pop(parent)

        for succ in successors.get(current, []):
            indegree[succ] -= edge_counts.get((current, succ), 1)
            if indegree[succ] == 0:
                ready.append(succ)

    return [graph.get_node(name) for name in order]


def evaluate_orders(graph: Graph, strategies: Iterable[Tuple[str, Callable[[Graph], Sequence[Node]]]] | None = None) -> List[OrderStats]:
    strategies = list(strategies) if strategies is not None else [
        ("default_topological", default_topological_order),
        ("greedy_min_cutwidth", greedy_min_cutwidth_order),
    ]
    stats: List[OrderStats] = []
    for name, fn in strategies:
        order_nodes = list(fn(graph))
        width = compute_frontier_width(graph, order_nodes)
        stats.append(OrderStats(strategy=name, order=order_nodes, max_live=width))
    stats.sort(key=lambda s: (s.max_live, s.strategy))
    return stats


def best_heuristic_order(graph: Graph, strategies: Iterable[Tuple[str, Callable[[Graph], Sequence[Node]]]] | None = None) -> OrderStats:
    stats = evaluate_orders(graph, strategies)
    if not stats:
        raise ValueError("No strategies provided for best_heuristic_order.")
    return stats[0]
