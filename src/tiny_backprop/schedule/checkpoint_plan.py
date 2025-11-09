from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from tiny_backprop.analysis.frontier_width import summarize_frontier
from tiny_backprop.analysis.width_heuristics import best_heuristic_order
from tiny_backprop.graph.ir import Graph
from tiny_backprop.schedule.block_partition import Block, partition_into_blocks
from tiny_backprop.schedule.interval_tree import IntervalNode, build_balanced_interval_tree


@dataclass
class CheckpointPlan:
    """
    Declarative description of where to store forward activations.

    Attributes:
        save_nodes: node indices (or names) to checkpoint.
        interval_tree: tree describing hierarchical decomposition.
    """
    save_nodes: List[int] = field(default_factory=list)
    interval_tree: Optional[IntervalNode] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def make_naive_plan(num_nodes: int) -> CheckpointPlan:
    """
    Trivial baseline: save every node (no recompute).
    """
    return CheckpointPlan(save_nodes=list(range(num_nodes)))


def _select_checkpoints(blocks: Sequence[Block]) -> List[int]:
    checkpoints: List[int] = []
    for block in blocks:
        checkpoints.append(block.start_idx)
    if blocks:
        checkpoints.append(blocks[-1].end_idx - 1)
    return sorted(set(idx for idx in checkpoints if idx >= 0))


def make_interval_plan(graph: Graph, *, block_size: int) -> CheckpointPlan:
    """
    Build a checkpoint plan by:
    1. Selecting a topological order (using heuristics).
    2. Partitioning into fixed-size blocks.
    3. Building a balanced interval tree over blocks.
    4. Saving boundary nodes of each block (coarse-grained checkpoints).
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    order_stats = best_heuristic_order(graph)
    order_nodes = order_stats.order

    blocks = partition_into_blocks(order_nodes, block_size=block_size)
    tree = build_balanced_interval_tree(blocks)

    checkpoints = _select_checkpoints(blocks)
    summary = summarize_frontier(graph, order_nodes)

    return CheckpointPlan(
        save_nodes=checkpoints,
        interval_tree=tree,
        meta={
            "order": [node.name for node in order_nodes],
            "block_size": block_size,
            "num_blocks": len(blocks),
            "peak_frontier": summary.max_live,
        },
    )
