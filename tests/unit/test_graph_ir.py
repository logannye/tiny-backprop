from __future__ import annotations

import pytest

from tiny_backprop.graph.ir import Graph, Node


def test_graph_topological_sort_and_frontier() -> None:
    graph = Graph()
    graph.add_node(Node(name="input", op="input", outputs_size=4))
    graph.inputs.append("input")
    graph.add_node(Node(name="linear", op="linear", inputs=["input"], outputs_size=8))
    graph.add_node(Node(name="relu", op="relu", inputs=["linear"], outputs_size=8))
    graph.outputs = ["relu"]

    graph.validate()

    order = [node.name for node in graph.topological_sort()]
    assert order == ["input", "linear", "relu"]

    frontier = graph.live_frontier_sizes()
    assert frontier == [4, 12, 16]


def test_graph_validate_missing_predecessor() -> None:
    graph = Graph()
    graph.add_node(Node(name="a", op="noop", inputs=["missing"], outputs_size=0))
    with pytest.raises(ValueError):
        graph.validate()


def test_topological_sort_detects_cycle() -> None:
    graph = Graph()
    graph.add_node(Node(name="a", op="noop", inputs=["c"], outputs_size=1))
    graph.add_node(Node(name="b", op="noop", inputs=["a"], outputs_size=1))
    graph.add_node(Node(name="c", op="noop", inputs=["b"], outputs_size=1))

    with pytest.raises(ValueError):
        graph.topological_sort()


def test_node_rejects_negative_activation() -> None:
    with pytest.raises(ValueError):
        Node(name="neg", op="noop", outputs_size=-1)

