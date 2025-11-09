from __future__ import annotations

import pytest

from tiny_backprop.graph.ir import Node
from tiny_backprop.schedule.block_partition import Block, partition_into_blocks


def test_partition_into_blocks_even() -> None:
    order = [Node(name=f"n{i}", op="noop") for i in range(6)]
    blocks = partition_into_blocks(order, block_size=2)

    assert [b.start_idx for b in blocks] == [0, 2, 4]
    assert [b.end_idx for b in blocks] == [2, 4, 6]


def test_partition_into_blocks_uneven() -> None:
    order = [Node(name=f"n{i}", op="noop") for i in range(5)]
    blocks = partition_into_blocks(order, block_size=3)

    assert len(blocks) == 2
    assert blocks[0] == Block(start_idx=0, end_idx=3)
    assert blocks[1] == Block(start_idx=3, end_idx=5)


def test_partition_into_blocks_raises_on_nonpositive_block_size() -> None:
    order = [Node(name="n0", op="noop")]
    with pytest.raises(ValueError):
        partition_into_blocks(order, block_size=0)

