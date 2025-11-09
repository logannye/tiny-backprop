from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from tiny_backprop.schedule.checkpoint_plan import CheckpointPlan
from tiny_backprop.schedule.interval_tree import Interval, IntervalNode


@dataclass
class ReplayStep:
    """
    A single step of the backward/recompute schedule.
    """
    kind: str  # "forward", "backward"
    start_idx: int
    end_idx: int


@dataclass
class ReplayPlan:
    """
    Ordered sequence of recompute/backward actions.
    """
    steps: List[ReplayStep] = field(default_factory=list)

    def add_step(self, kind: str, interval: Interval) -> None:
        self.steps.append(
            ReplayStep(kind=kind, start_idx=interval.start, end_idx=interval.end)
        )

    def __iter__(self):
        return iter(self.steps)


def _post_order(node: Optional[IntervalNode]) -> Iterable[IntervalNode]:
    if node is None:
        return
    if node.left:
        yield from _post_order(node.left)
    if node.right:
        yield from _post_order(node.right)
    yield node


def build_replay_plan(plan: CheckpointPlan) -> ReplayPlan:
    """
    Placeholder: convert interval_tree + save_nodes into an actual schedule.
    """
    replay = ReplayPlan()
    for node in _post_order(plan.interval_tree):
        if node.is_leaf():
            replay.add_step("backward", node.interval)
        else:
            replay.add_step("forward", node.interval)
            replay.add_step("backward", node.interval)
    return replay
