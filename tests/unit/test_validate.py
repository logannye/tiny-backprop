from __future__ import annotations

import pytest

from tiny_backprop.graph.ir import Graph, Node
from tiny_backprop.schedule.checkpoint_plan import make_interval_plan
from tiny_backprop.schedule.replay_plan import ReplayPlan, ReplayStep, build_replay_plan
from tiny_backprop.schedule.validate import validate_plan


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


def test_validate_plan_accepts_consistent_schedule() -> None:
    graph = _build_chain_graph(4)
    plan = make_interval_plan(graph, block_size=2)
    replay = build_replay_plan(plan)
    validate_plan(plan, replay)


def test_validate_plan_rejects_mismatched_replay() -> None:
    graph = _build_chain_graph(4)
    plan = make_interval_plan(graph, block_size=2)
    bad_replay = ReplayPlan(steps=[ReplayStep(kind="backward", start_idx=0, end_idx=1)])

    with pytest.raises(ValueError):
        validate_plan(plan, bad_replay)

