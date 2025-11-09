from __future__ import annotations

from tiny_backprop.graph.ir import Graph, Node
from tiny_backprop.schedule.checkpoint_plan import (
    CheckpointPlan,
    make_interval_plan,
    make_naive_plan,
)


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


def test_make_naive_plan_saves_all_nodes() -> None:
    plan = make_naive_plan(4)
    assert isinstance(plan, CheckpointPlan)
    assert plan.save_nodes == [0, 1, 2, 3]
    assert plan.interval_tree is None


def test_make_interval_plan_creates_tree_and_metadata() -> None:
    graph = _build_chain_graph(6)
    plan = make_interval_plan(graph, block_size=2)

    assert plan.interval_tree is not None
    leaves = list(plan.interval_tree.iter_leaves())
    assert len(leaves) == 3
    assert plan.save_nodes  # contains boundary indices
    assert plan.meta["block_size"] == 2
    assert plan.meta["num_blocks"] == 3
    assert plan.meta["order"] == [f"n{i}" for i in range(6)]

