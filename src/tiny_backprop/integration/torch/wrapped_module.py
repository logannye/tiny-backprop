from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch
from torch import nn

from tiny_backprop.integration.torch.fx_capture import capture_graph
from tiny_backprop.runtime.executor import ExecutionCallbacks, HeightCompressedExecutor

PlannerFn = Callable[[nn.Module, Tuple[Any, ...]], HeightCompressedExecutor]


@dataclass
class HCConfig:
    block_size: Optional[int] = None
    eager_plan: bool = False
    planner: Optional[PlannerFn] = None


class HCModule(nn.Module):
    """
    Drop-in wrapper that orchestrates height-compressed planning for PyTorch models.

    Example::

        model = HCModule(MyNetwork(), config=HCConfig(block_size=16))
        out = model(x)
        loss = loss_fn(out, y)
        model.backward(loss)

    The executor presently defaults to autograd execution but pre-computes the
    scheduling metadata so integration layers can inspect memory savings.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        config: Optional[HCConfig] = None,
    ) -> None:
        super().__init__()
        self.module = module
        self.config = config or HCConfig()
        self._planner = self.config.planner
        self._executor: Optional[HeightCompressedExecutor] = None
        self._latest_inputs: Optional[Tuple[Any, ...]] = None

        if self.config.eager_plan:
            raise ValueError(
                "Eager planning requires example inputs; call `configure_executor` explicitly."
            )

    def forward(self, *args, **kwargs) -> Any:
        self._latest_inputs = self._normalize_inputs(args, kwargs)
        return self.module(*args, **kwargs)

    def configure_executor(
        self, example_inputs: Optional[Tuple[Any, ...]] = None
    ) -> HeightCompressedExecutor:
        inputs = example_inputs or self._latest_inputs
        if inputs is None:
            raise ValueError(
                "Cannot configure executor without example inputs. "
                "Run a forward pass or pass `example_inputs`."
            )

        if self._planner is not None:
            executor = self._planner(self.module, inputs)
        else:
            executor = self._default_planner(inputs)

        self._executor = executor
        return executor

    def backward(
        self,
        loss: torch.Tensor,
        *,
        callbacks: Optional[ExecutionCallbacks] = None,
    ) -> None:
        executor = self._executor or self.configure_executor()
        executor.backward(
            loss,
            autograd_backward=torch.autograd.backward,
            callbacks=callbacks,
        )

    @property
    def executor(self) -> Optional[HeightCompressedExecutor]:
        return self._executor

    def _default_planner(self, inputs: Tuple[Any, ...]) -> HeightCompressedExecutor:
        try:
            graph = capture_graph(self.module, example_inputs=inputs)
        except Exception:
            return HeightCompressedExecutor.naive(num_nodes=0)

        executor = HeightCompressedExecutor()
        try:
            executor.configure(graph, block_size=self.config.block_size)
        except Exception:
            return HeightCompressedExecutor.naive(num_nodes=len(graph.nodes))
        return executor

    @staticmethod
    def _normalize_inputs(args, kwargs) -> Tuple[Any, ...]:
        if kwargs:
            ordered_kwargs = tuple(value for _, value in sorted(kwargs.items()))
            return tuple(args) + ordered_kwargs
        return tuple(args)
