"""
Hooks to wrap JAX grad/jit with height-compressed backprop.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from tiny_backprop.integration.jax.jaxpr_capture import capture_graph_jaxpr
from tiny_backprop.runtime.executor import HeightCompressedExecutor

try:
    import jax
except ModuleNotFoundError:  # pragma: no cover - handled in tests
    jax = None  # type: ignore


GradFn = Callable[..., Any]


def height_compressed_grad(
    fn: Callable[..., Any],
    *,
    block_size: Optional[int] = None,
) -> GradFn:
    """
    Wrap ``jax.grad`` so that each invocation also plans a height-compressed
    schedule. The returned gradient function exposes ``last_executor`` for
    inspection.
    """
    if jax is None:  # pragma: no cover
        raise ModuleNotFoundError("JAX is required for height_compressed_grad.")

    grad_impl = jax.grad(fn)

    def wrapped(*args, **kwargs):
        executor: Optional[HeightCompressedExecutor] = None
        try:
            graph = capture_graph_jaxpr(fn, example_inputs=args)
            executor = HeightCompressedExecutor()
            executor.configure(graph, block_size=block_size)
        except Exception:
            executor = None
        wrapped.last_executor = executor  # type: ignore[attr-defined]
        return grad_impl(*args, **kwargs)

    wrapped.last_executor = None  # type: ignore[attr-defined]
    wrapped.__name__ = f"height_compressed_grad_{getattr(fn, '__name__', 'fn')}"
    return wrapped
