"""
FX-based capture of PyTorch computation graphs.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.fx as fx

from tiny_backprop.graph.builders import from_pytorch_fx
from tiny_backprop.graph.ir import Graph


def _ensure_tuple(example_inputs: Any) -> Tuple[Any, ...]:
    if isinstance(example_inputs, tuple):
        return example_inputs
    if isinstance(example_inputs, list):
        return tuple(example_inputs)
    return (example_inputs,)


def capture_graph(module: torch.nn.Module, example_inputs: Any) -> Graph:
    """
    Trace module with FX and build our Graph.

    Args:
        module: The PyTorch module to trace.
        example_inputs: Sample inputs used both for tracing and shape
            propagation. Accepts a single tensor, tuple/list of tensors,
            or arbitrary nested structures supported by FX.

    Returns:
        Graph: Framework-agnostic DAG populated with activation sizes derived
        from FX tensor metadata.
    """
    traced = fx.symbolic_trace(module)
    example_tuple = _ensure_tuple(example_inputs)
    graph = from_pytorch_fx(traced, example_inputs=example_tuple)
    graph.metadata.setdefault("module_repr", repr(module))
    graph.metadata.setdefault("module_type", module.__class__.__qualname__)
    return graph
