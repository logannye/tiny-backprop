from __future__ import annotations

from tiny_backprop.analysis.frontier_width import compute_frontier_width
from tiny_backprop.analysis.width_heuristics import (
    best_heuristic_order,
    evaluate_orders,
    greedy_min_cutwidth_order,
)
from tiny_backprop.graph.ir import Graph, Node


def _build_branchy_graph() -> Graph:
    graph = Graph()
    graph.add_node(Node(name="input_a", op="input", outputs_size=4))
    graph.add_node(Node(name="input_b", op="input", outputs_size=100))
    graph.add_node(Node(name="use_a", op="linear", inputs=["input_a"], outputs_size=5))
    graph.add_node(Node(name="use_b", op="linear", inputs=["input_b"], outputs_size=5))
    graph.add_node(
        Node(name="join", op="concat", inputs=["use_a", "use_b"], outputs_size=8)
    )
    graph.inputs = ["input_a", "input_b"]
    graph.outputs = ["join"]
    graph.validate()
    return graph


def test_greedy_min_cutwidth_reorders_to_reduce_frontier() -> None:
    graph = _build_branchy_graph()
    greedy_order = greedy_min_cutwidth_order(graph)
    greedy_names = [node.name for node in greedy_order]

    assert greedy_names == ["input_a", "use_a", "input_b", "use_b", "join"]

    default_width = compute_frontier_width(graph)
    greedy_width = compute_frontier_width(graph, greedy_order)
    assert greedy_width <= default_width


def test_evaluate_orders_returns_sorted_stats() -> None:
    graph = _build_branchy_graph()
    stats = evaluate_orders(graph)

    assert stats[0].max_live <= stats[-1].max_live
    assert stats[0].strategy in {"greedy_min_cutwidth", "default_topological"}
    assert len(stats[0].order) == len(graph.nodes)


def test_best_heuristic_order_matches_first_stat() -> None:
    graph = _build_branchy_graph()
    best = best_heuristic_order(graph)
    stats = evaluate_orders(graph)
    assert best.strategy == stats[0].strategy
    assert [node.name for node in best.order] == [node.name for node in stats[0].order]

