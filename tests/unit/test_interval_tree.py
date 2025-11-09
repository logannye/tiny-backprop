from __future__ import annotations

from tiny_backprop.graph.ir import Node
from tiny_backprop.schedule.block_partition import partition_into_blocks
from tiny_backprop.schedule.interval_tree import (
    Interval,
    IntervalNode,
    build_balanced_interval_tree,
    flatten_intervals,
)


def _make_blocks(num_nodes: int, block_size: int):
    order = [Node(name=f"n{i}", op="noop") for i in range(num_nodes)]
    return partition_into_blocks(order, block_size)


def test_build_balanced_interval_tree() -> None:
    blocks = _make_blocks(8, 2)
    root = build_balanced_interval_tree(blocks)

    assert isinstance(root, IntervalNode)
    assert root.interval == Interval(0, 8)
    assert root.depth() == 3
    leaves = list(root.iter_leaves())
    assert len(leaves) == 4
    assert leaves[0].interval == Interval(0, 2)


def test_flatten_intervals_returns_all_nodes() -> None:
    blocks = _make_blocks(6, 2)
    root = build_balanced_interval_tree(blocks)
    intervals = flatten_intervals(root)
    lengths = sorted(interval.end - interval.start for interval in intervals)
    assert lengths[:3] == [2, 2, 2]

