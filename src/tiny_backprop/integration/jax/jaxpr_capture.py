from __future__ import annotations

from typing import Any, Sequence

from tiny_backprop.graph.builders import from_jaxpr
from tiny_backprop.graph.ir import Graph


def capture_graph_jaxpr(fn, example_inputs: Sequence[Any]) -> Graph:
    """
    Trace ``fn`` with ``jax.make_jaxpr`` and convert to the internal Graph.

    Args:
        fn: Callable to trace.
        example_inputs: Sequence of sample inputs. These are passed directly to
            ``jax.make_jaxpr``; therefore they must be JAX-compatible values.

    Returns:
        Graph capturing the forward DAG implied by the JAXPR.
    """
    try:
        import jax
    except ModuleNotFoundError as exc:  # pragma: no cover - handled in tests
        raise ModuleNotFoundError(
            "capture_graph_jaxpr requires JAX to be installed."
        ) from exc

    if not isinstance(example_inputs, (list, tuple)):
        example_inputs = (example_inputs,)

    closed_jaxpr = jax.make_jaxpr(fn)(*example_inputs)
    graph = from_jaxpr(closed_jaxpr)
    graph.metadata.setdefault("function_name", getattr(fn, "__name__", "<lambda>"))
    return graph
