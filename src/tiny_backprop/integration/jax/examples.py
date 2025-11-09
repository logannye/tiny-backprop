"""
Demonstrate the JAX integration surface.
"""

from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover
    jax = None  # type: ignore
    jnp = None  # type: ignore

from tiny_backprop.integration.jax.transformations import height_compressed_grad


def demo() -> None:
    if jax is None:  # pragma: no cover
        print("JAX is not available; install jax to run the demo.")
        return

    def loss_fn(params, x):
        w, b = params
        hidden = jnp.tanh(x @ w + b)
        return jnp.mean(hidden**2)

    x = jnp.ones((4, 3))
    hc_grad = height_compressed_grad(lambda params: loss_fn(params, x), block_size=2)

    params = (
        jnp.ones((3, 5)),
        jnp.zeros((5,)),
    )
    grads = hc_grad(params)
    executor = hc_grad.last_executor  # type: ignore[attr-defined]

    print("Gradient norms:", [float(jnp.linalg.norm(g)) for g in grads])
    if executor and executor.checkpoint_plan:
        print("Planned blocks:", executor.checkpoint_plan.meta.get("num_blocks"))


if __name__ == "__main__":
    demo()
