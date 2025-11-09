"""
Lightweight profiling hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ProfileStats:
    peak_memory_bytes: int = 0
    recompute_flops: float = 0.0
    recompute_steps: int = 0
    events: Dict[str, float] = field(default_factory=dict)


class Profiler:
    def __init__(self) -> None:
        self.stats = ProfileStats()

    def record_memory(self, bytes_used: int) -> None:
        if bytes_used > self.stats.peak_memory_bytes:
            self.stats.peak_memory_bytes = bytes_used

    def record_flops(self, flops: float) -> None:
        self.stats.recompute_flops += flops
        self.stats.recompute_steps += 1

    def record_event(self, name: str, duration_ms: float) -> None:
        self.stats.events[name] = self.stats.events.get(name, 0.0) + duration_ms

    def snapshot(self) -> ProfileStats:
        return self.stats
