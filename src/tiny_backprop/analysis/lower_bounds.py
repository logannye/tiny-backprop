"""
Pebbling-style / cutwidth-style lower bound estimates on memory.
"""

from __future__ import annotations

from tiny_backprop.analysis.frontier_width import compute_frontier_width
from tiny_backprop.analysis.width_heuristics import evaluate_orders
from tiny_backprop.graph.ir import Graph


def naive_lower_bound(graph: Graph) -> int:
    """
    Trivial bound: must at least hold max single activation.
    """
    return max((n.outputs_size for n in graph.nodes.values()), default=0)


def frontier_based_lower_bound(graph: Graph) -> int:
    """
    Currently just returns frontier width of default order.

    Later:
    - compute min over multiple heuristics
    - use this as "target" for near-optimal schedules.
    """
    return compute_frontier_width(graph)


def lower_bound_report(graph: Graph) -> dict[str, int]:
    """
    Aggregate a set of easily computed lower bounds that planners can compare
    against when evaluating schedules.
    """
    bounds: dict[str, int] = {
        "max_activation": naive_lower_bound(graph),
    }

    heuristic_orders = evaluate_orders(graph)
    if heuristic_orders:
        bounds["heuristic_frontier"] = min(stat.max_live for stat in heuristic_orders)
    else:
        bounds["heuristic_frontier"] = 0

    bounds["default_frontier"] = compute_frontier_width(graph)
    return bounds
