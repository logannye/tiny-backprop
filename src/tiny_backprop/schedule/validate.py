"""
Validation of checkpoint & replay plans.
"""

from __future__ import annotations

from tiny_backprop.schedule.checkpoint_plan import CheckpointPlan
from tiny_backprop.schedule.interval_tree import IntervalNode
from tiny_backprop.schedule.replay_plan import ReplayPlan


def validate_plan(plan: CheckpointPlan, replay: ReplayPlan) -> None:
    """
    Placeholder for structural validation:
    - all needed activations are available or recomputable
    - no inconsistent indices
    """
    _ensure_checkpoint_indices_sorted(plan)
    _ensure_interval_tree_consistency(plan.interval_tree)
    _ensure_replay_matches_tree(plan, replay)


def _ensure_checkpoint_indices_sorted(plan: CheckpointPlan) -> None:
    if plan.save_nodes != sorted(plan.save_nodes):
        raise ValueError("Checkpoint indices must be sorted.")


def _ensure_interval_tree_consistency(node: IntervalNode | None) -> None:
    if node is None:
        return
    if node.interval.start >= node.interval.end:
        raise ValueError("Invalid interval with non-positive length.")
    if node.left:
        left = node.left.interval
        if left.start != node.interval.start or left.end > node.interval.end:
            raise ValueError("Left child interval not contained within parent.")
        _ensure_interval_tree_consistency(node.left)
    if node.right:
        right = node.right.interval
        if right.end != node.interval.end or right.start < node.interval.start:
            raise ValueError("Right child interval not contained within parent.")
        if node.left and node.left.interval.end > right.start:
            raise ValueError("Overlapping child intervals detected.")
        _ensure_interval_tree_consistency(node.right)


def _ensure_replay_matches_tree(plan: CheckpointPlan, replay: ReplayPlan) -> None:
    tree_intervals = []
    if plan.interval_tree:
        for node in plan.interval_tree.iter_nodes():
            tree_intervals.append((node.interval.start, node.interval.end, node.is_leaf()))
    if not tree_intervals and replay.steps:
        raise ValueError("Replay plan has steps but checkpoint plan has no intervals.")

    expected_forward = [
        (start, end)
        for start, end, is_leaf in tree_intervals
        if not is_leaf
    ]
    expected_backward = [
        (start, end)
        for start, end, _ in tree_intervals
    ]

    forward_steps = [
        (step.start_idx, step.end_idx) for step in replay.steps if step.kind == "forward"
    ]
    backward_steps = [
        (step.start_idx, step.end_idx) for step in replay.steps if step.kind == "backward"
    ]

    if sorted(forward_steps) != sorted(expected_forward):
        raise ValueError("Replay plan forward intervals mismatch interval tree.")
    if sorted(backward_steps) != sorted(expected_backward):
        raise ValueError("Replay plan backward intervals mismatch interval tree.")
