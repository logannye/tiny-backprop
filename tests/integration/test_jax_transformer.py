from __future__ import annotations

import pytest

pytest.importorskip("jax")
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore

from tiny_backprop.integration.jax.transformations import height_compressed_grad


def attention_like(params, query):
    w_q, w_k, w_v = params
    q = query @ w_q
    k = query @ w_k
    v = query @ w_v
    scores = jnp.matmul(q, k.T) / jnp.sqrt(q.shape[-1])
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.sum(weights @ v)


def test_height_compressed_grad_on_attention() -> None:
    block_size = 3
    grad_fn = height_compressed_grad(
        lambda params: attention_like(params, jnp.ones((4, 4))),
        block_size=block_size,
    )

    params = (
        jnp.ones((4, 4)) * 0.1,
        jnp.ones((4, 4)) * 0.2,
        jnp.ones((4, 4)) * 0.3,
    )

    grads = grad_fn(params)
    executor = grad_fn.last_executor  # type: ignore[attr-defined]

    ref_grads = jax.grad(lambda p: attention_like(p, jnp.ones((4, 4))))(params)

    assert all(jnp.allclose(g, r, atol=1e-5) for g, r in zip(grads, ref_grads))
    assert executor is not None
    assert executor.checkpoint_plan is not None
    assert executor.checkpoint_plan.meta.get("block_size") == block_size

