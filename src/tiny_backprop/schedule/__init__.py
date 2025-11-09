"""
Schedule construction for height-compressed backprop.

This package turns a computation graph into:
- A block partition (coarse-grained units of work).
- A hierarchical interval tree (for height compression).
- A checkpoint plan (what to save).
- A replay plan (what to recompute, in what order).
"""

from .block_partition import Block, partition_into_blocks
from .interval_tree import IntervalNode, build_interval_tree
from .checkpoint_plan import CheckpointPlan, make_naive_plan
from .replay_plan import ReplayPlan, ReplayStep, build_replay_plan
from .validate import validate_plan

__all__ = [
    "Block",
    "partition_into_blocks",
    "IntervalNode",
    "build_interval_tree",
    "CheckpointPlan",
    "make_naive_plan",
    "ReplayPlan",
    "ReplayStep",
    "build_replay_plan",
    "validate_plan",
]
