from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from tiny_backprop.graph.ir import Graph
from tiny_backprop.runtime.hooks import ActivationRegistry
from tiny_backprop.runtime.profiling import Profiler
from tiny_backprop.runtime.storage import CheckpointStorage
from tiny_backprop.schedule.checkpoint_plan import (
    CheckpointPlan,
    make_interval_plan,
    make_naive_plan,
)
from tiny_backprop.schedule.replay_plan import ReplayPlan, build_replay_plan
from tiny_backprop.schedule.validate import validate_plan


@dataclass
class ExecutionCallbacks:
    """
    Callbacks supplied by framework integrations to execute replay steps.

    Args:
        run_forward: Recompute forward activations for nodes in [start, end).
        run_backward: Execute backward pass for nodes in [start, end).
        prepare_loss: Optional hook before replay begins (e.g., seed grads).
        finalize: Optional hook after replay completes.
    """

    run_forward: Callable[[int, int], None]
    run_backward: Callable[[int, int], None]
    prepare_loss: Optional[Callable[[Any], None]] = None
    finalize: Optional[Callable[[], None]] = None


class HeightCompressedExecutor:
    """
    Executes forward/backward according to a checkpoint & replay plan.

    This is the abstraction that framework integrations will use.
    """

    def __init__(
        self,
        checkpoint_plan: Optional[CheckpointPlan] = None,
        replay_plan: Optional[ReplayPlan] = None,
    ) -> None:
        self.checkpoint_plan = checkpoint_plan
        self.replay_plan = replay_plan
        self.storage = CheckpointStorage()
        self.activations = ActivationRegistry()
        self.profiler = Profiler()

    @classmethod
    def naive(cls, num_nodes: int) -> "HeightCompressedExecutor":
        plan = make_naive_plan(num_nodes)
        replay = build_replay_plan(plan)
        return cls(plan, replay)

    def configure(self, graph: Graph, *, block_size: Optional[int] = None) -> None:
        """
        Generate checkpoint and replay plans for the given graph.

        Args:
            graph: Graph to analyse and schedule.
            block_size: Optional block size override. Defaults to ceil(sqrt(N)).
        """
        if block_size is None:
            block_size = max(1, int(len(graph.nodes) ** 0.5))

        if block_size <= 0:
            plan = make_naive_plan(len(graph.nodes))
        else:
            plan = make_interval_plan(graph, block_size=block_size)

        replay = build_replay_plan(plan)
        validate_plan(plan, replay)

        self.checkpoint_plan = plan
        self.replay_plan = replay
        self.storage.clear()
        self.activations.clear()
        self.profiler = Profiler()

    def execute(self, callbacks: ExecutionCallbacks) -> None:
        if self.replay_plan is None:
            raise ValueError("Replay plan has not been configured.")

        if callbacks.prepare_loss:
            callbacks.prepare_loss(None)

        for step in self.replay_plan:
            if step.kind == "forward":
                callbacks.run_forward(step.start_idx, step.end_idx)
            elif step.kind == "backward":
                callbacks.run_backward(step.start_idx, step.end_idx)
            else:
                raise ValueError(f"Unknown replay step kind: {step.kind}")

        if callbacks.finalize:
            callbacks.finalize()

    def backward(
        self,
        loss: Any,
        autograd_backward: Callable[[Any], None],
        callbacks: Optional[ExecutionCallbacks] = None,
    ) -> None:
        """
        Execute backward according to the configured replay plan. If callbacks
        are not supplied, fall back to standard autograd.
        """
        if callbacks is None or self.replay_plan is None:
            autograd_backward(loss)
            return

        if callbacks.prepare_loss:
            callbacks.prepare_loss(loss)

        for step in self.replay_plan:
            if step.kind == "forward":
                callbacks.run_forward(step.start_idx, step.end_idx)
            elif step.kind == "backward":
                callbacks.run_backward(step.start_idx, step.end_idx)
            else:
                raise ValueError(f"Unknown replay step kind: {step.kind}")

        if callbacks.finalize:
            callbacks.finalize()
