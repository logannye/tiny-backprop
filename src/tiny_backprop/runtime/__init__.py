"""
Runtime support for executing planned height-compressed schedules.

This layer is responsible for:
- Driving the forward pass with checkpointing.
- Replaying segments as needed during backward.
- Interfacing with framework autograd (Torch, JAX, etc.).
"""

from .executor import ExecutionCallbacks, HeightCompressedExecutor
from .storage import CheckpointStorage
from .hooks import ActivationRegistry
from .profiling import Profiler, ProfileStats

__all__ = [
    "ExecutionCallbacks",
    "HeightCompressedExecutor",
    "CheckpointStorage",
    "ActivationRegistry",
    "Profiler",
    "ProfileStats",
]
