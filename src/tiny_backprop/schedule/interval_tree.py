from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from tiny_backprop.schedule.block_partition import Block


@dataclass(frozen=True)
class Interval:
    start: int
    end: int  # exclusive

    def size(self) -> int:
        return self.end - self.start

    def contains(self, idx: int) -> bool:
        return self.start <= idx < self.end


@dataclass
class IntervalNode:
    interval: Interval
    left: Optional["IntervalNode"] = None
    right: Optional["IntervalNode"] = None

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def depth(self) -> int:
        if self.is_leaf():
            return 1
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return 1 + max(left_depth, right_depth)

    def iter_leaves(self) -> Iterable["IntervalNode"]:
        if self.is_leaf():
            yield self
        else:
            if self.left:
                yield from self.left.iter_leaves()
            if self.right:
                yield from self.right.iter_leaves()

    def iter_nodes(self) -> Iterable["IntervalNode"]:
        yield self
        if self.left:
            yield from self.left.iter_nodes()
        if self.right:
            yield from self.right.iter_nodes()


def build_balanced_interval_tree(blocks: List[Block]) -> Optional[IntervalNode]:
    """
    Build a balanced binary tree over contiguous blocks. Each leaf corresponds
    to a block; internal nodes span consecutive blocks.
    """
    if not blocks:
        return None

    leaves = [IntervalNode(interval=Interval(b.start_idx, b.end_idx)) for b in blocks]

    while len(leaves) > 1:
        next_level: List[IntervalNode] = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            if i + 1 < len(leaves):
                right = leaves[i + 1]
                interval = Interval(start=left.interval.start, end=right.interval.end)
                next_level.append(IntervalNode(interval=interval, left=left, right=right))
            else:
                next_level.append(left)
        leaves = next_level

    return leaves[0]


def build_interval_tree(blocks: List[Block]) -> Optional[IntervalNode]:
    """
    Backwards-compatible alias for build_balanced_interval_tree.
    """
    return build_balanced_interval_tree(blocks)


def flatten_intervals(node: Optional[IntervalNode]) -> List[Interval]:
    if node is None:
        return []
    return [n.interval for n in node.iter_nodes()]

