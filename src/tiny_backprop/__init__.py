"""
tiny-backprop

Height-compressed backpropagation for arbitrary computation graphs.
"""

from .graph.ir import Graph, Node
from .schedule.checkpoint_plan import CheckpointPlan
from .schedule.replay_plan import ReplayPlan
from .runtime.executor import HeightCompressedExecutor

__all__ = [
    "Graph",
    "Node",
    "CheckpointPlan",
    "ReplayPlan",
    "HeightCompressedExecutor",
]
