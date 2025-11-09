from __future__ import annotations

from tiny_backprop.graph.ir import Graph, Node
from tiny_backprop.schedule.checkpoint_plan import make_interval_plan
from tiny_backprop.schedule.replay_plan import ReplayPlan, build_replay_plan


def _build_chain_graph(num_nodes: int) -> Graph:
    graph = Graph()
    prev = None
    for idx in range(num_nodes):
        name = f"n{idx}"
        inputs = [prev] if prev is not None else []
        graph.add_node(Node(name=name, op="linear", inputs=inputs, outputs_size=idx + 1))
        if prev is None:
            graph.inputs = [name]
        prev = name
    graph.outputs = [prev] if prev else []
    graph.validate()
    return graph


def test_build_replay_plan_produces_postorder_schedule() -> None:
    graph = _build_chain_graph(4)
    plan = make_interval_plan(graph, block_size=2)

    replay = build_replay_plan(plan)
    assert isinstance(replay, ReplayPlan)
    steps = [(step.kind, step.start_idx, step.end_idx) for step in replay]
    assert steps == [
        ("backward", 0, 2),
        ("backward", 2, 4),
        ("forward", 0, 4),
        ("backward", 0, 4),
    ]

