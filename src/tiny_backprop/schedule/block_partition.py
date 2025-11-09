from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from tiny_backprop.graph.ir import Node


@dataclass
class Block:
    start_idx: int
    end_idx: int  # exclusive


def partition_into_blocks(order: Iterable[Node], block_size: int) -> List[Block]:
    """
    Partition topo-ordered nodes into contiguous blocks.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    order_list = list(order)
    blocks: List[Block] = []
    n = len(order_list)
    i = 0
    while i < n:
        j = min(i + block_size, n)
        blocks.append(Block(start_idx=i, end_idx=j))
        i = j
    return blocks
