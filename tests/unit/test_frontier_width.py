from __future__ import annotations

from tiny_backprop.analysis.frontier_width import (
    FrontierSnapshot,
    FrontierSummary,
    compute_frontier_width,
    frontier_profile,
    summarize_frontier,
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


def test_frontier_profile_records_entering_and_exiting_nodes() -> None:
    graph = _build_branchy_graph()
    profile = frontier_profile(graph)

    assert len(profile) == len(graph.nodes)
    second_step = profile[1]
    assert isinstance(second_step, FrontierSnapshot)
    assert second_step.node == "input_b"
    assert second_step.live_memory == 104
    assert second_step.entering == ("input_b",)
    assert second_step.exiting == ()

    third_step = profile[2]
    assert third_step.node == "use_a"
    assert third_step.live_memory == 105  # input_a freed after this step
    assert "input_a" in third_step.exiting


def test_summarize_frontier_matches_compute_frontier_width() -> None:
    graph = _build_branchy_graph()
    summary = summarize_frontier(graph)

    assert isinstance(summary, FrontierSummary)
    assert summary.max_live == compute_frontier_width(graph)
    assert summary.argmax_step == 2  # after processing use_a
    assert summary.final_live == 8

