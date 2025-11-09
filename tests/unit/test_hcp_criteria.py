from __future__ import annotations

from tiny_backprop.analysis.hcp_criteria import HCPEvaluation, evaluate_hcp
from tiny_backprop.graph.ir import Graph, Node


def _build_chain_graph(num_nodes: int, activation_size: int) -> Graph:
    graph = Graph()
    previous = None
    for idx in range(num_nodes):
        name = f"n{idx}"
        inputs = [previous] if previous else []
        graph.add_node(Node(name=name, op="linear", inputs=inputs, outputs_size=activation_size))
        if previous is None:
            graph.inputs = [name]
        previous = name
    graph.outputs = [previous] if previous else []
    graph.validate()
    return graph


def _build_wide_graph() -> Graph:
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


def test_evaluate_hcp_chain_graph_is_hcp_like() -> None:
    graph = _build_chain_graph(num_nodes=6, activation_size=10)
    result = evaluate_hcp(graph)

    assert isinstance(result, HCPEvaluation)
    assert result.is_hcp_like
    assert result.frontier_width <= 20
    assert result.live_ratio <= 0.35
    assert any("Frontier tail" in note for note in result.notes)


def test_evaluate_hcp_wide_graph_not_hcp_like() -> None:
    graph = _build_wide_graph()
    result = evaluate_hcp(graph)

    assert not result.is_hcp_like
    assert result.frontier_width > 0
    assert result.live_ratio > 0.35
    assert any("high" in note.lower() for note in result.notes)

