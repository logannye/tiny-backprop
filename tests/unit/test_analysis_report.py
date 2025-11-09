from __future__ import annotations

from tiny_backprop.analysis.report import GraphAnalysisReport, analyze_graph
from tiny_backprop.graph.ir import Graph, Node


def _build_chain_graph() -> Graph:
    graph = Graph()
    prev = None
    for idx in range(4):
        name = f"n{idx}"
        inputs = [prev] if prev is not None else []
        graph.add_node(Node(name=name, op="linear", inputs=inputs, outputs_size=10))
        if prev is None:
            graph.inputs = [name]
        prev = name
    graph.outputs = [prev] if prev is not None else []
    graph.validate()
    return graph


def test_analyze_graph_returns_report() -> None:
    graph = _build_chain_graph()
    report = analyze_graph(graph)

    assert isinstance(report, GraphAnalysisReport)
    assert report.primary_order == [f"n{i}" for i in range(4)]
    assert report.frontier.max_live == 10
    assert report.lower_bounds["max_activation"] == 10
    assert report.heuristic_orders
    assert report.hcp.is_hcp_like


def test_analyze_graph_without_snapshots() -> None:
    graph = _build_chain_graph()
    report = analyze_graph(graph, include_snapshots=False)
    assert report.frontier.snapshots == []

