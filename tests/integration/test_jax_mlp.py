from __future__ import annotations

import pytest

pytest.importorskip("jax")
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore

from tiny_backprop.integration.jax.jaxpr_capture import capture_graph_jaxpr
from tiny_backprop.integration.jax.transformations import height_compressed_grad


def simple_fn(x):
    hidden = jnp.sin(x)
    return jnp.tanh(hidden @ jnp.ones((x.shape[-1], 3)))


def test_jaxpr_capture_simple_fn() -> None:
    x = jnp.ones((2, 4))
    graph = capture_graph_jaxpr(simple_fn, example_inputs=(x,))

    assert graph.metadata["framework"] == "jaxpr"
    graph.validate()

    assert len(graph.inputs) == 1
    assert graph.outputs
    assert all(node.outputs_size >= 0 for node in graph.topological_sort())


def test_height_compressed_grad_matches_grad() -> None:
    def loss_fn(w):
        x = jnp.linspace(0, 1, 4)
        return jnp.sum(jnp.sin(x @ w))

    w = jnp.ones((4, 3))
    hc_grad = height_compressed_grad(loss_fn, block_size=2)
    grads_hc = hc_grad(w)
    grads_ref = jax.grad(loss_fn)(w)

    assert jnp.allclose(grads_hc, grads_ref, atol=1e-6)
    executor = hc_grad.last_executor  # type: ignore[attr-defined]
    assert executor is not None
    assert executor.checkpoint_plan is not None

