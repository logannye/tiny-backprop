# JAX Integration Guide

JAX support is experimental but already allows you to capture JAXPRs, inspect
height-compressibility, and plan schedules alongside gradient evaluations.

## Installation

```bash
pip install -e .
pip install jax jaxlib  # choose cuda or cpu build
```

Ensure you have a compatible version of JAX for your hardware (CPU-only builds do
not expose CUDA memory counters, but the planner still works).

## Capturing a JAXPR

```python
import jax
import jax.numpy as jnp
from tiny_backprop.integration.jax import capture_graph_jaxpr

def f(x):
    hidden = jnp.sin(x)
    return jnp.tanh(hidden @ jnp.ones((x.shape[-1], 3)))

x = jnp.ones((2, 4))
graph = capture_graph_jaxpr(f, example_inputs=(x,))
print(len(graph.nodes), "nodes captured")
```

Use the graph with `tiny_backprop.analysis.analyze_graph` to inspect frontier width
and near-optimality gaps.

## Height-Compressed Gradient Wrapper

```python
import jax.numpy as jnp
from tiny_backprop.integration.jax import height_compressed_grad

def loss_fn(params):
    w, b = params
    hidden = jnp.tanh(jnp.dot(w, b))
    return jnp.sum(hidden**2)

hc_grad = height_compressed_grad(loss_fn, block_size=16)
params = (
    jnp.ones((8, 8)),
    jnp.ones((8,)),
)
grads = hc_grad(params)
executor = hc_grad.last_executor  # inspect the planned schedule
```

- If planning succeeds, `last_executor` holds the configured
  `HeightCompressedExecutor`.
- On failure (e.g., unsupported primitive), the wrapper gracefully falls back to
  native `jax.grad` and sets `last_executor` to `None`.

## Inspecting Plans

```python
report = executor.checkpoint_plan.meta
print("Block size:", report["block_size"])
print("Peak frontier (bytes):", report["peak_frontier"])
```

You can also extract detailed frontier snapshots via
`tiny_backprop.analysis.summarize_frontier`.

## Limitations & Roadmap

- Only reverse-mode gradients (`jax.grad`) are wrapped today. `jit`, `pmap`, and
  higher-order transformations will be layered on once the runtime supports
  checkpoint storage across asynchronous devices.
- The wrapper currently assumes static shapes. If your function produces shape
  polymorphism, you may need to annotate or use `jax.experimental.shapecheck` before
  planning.
- Memory reporting is limited to peak activation estimates (JAX does not expose
  real-time allocator stats yet).

## Benchmarking

The existing benchmarks are PyTorch-based. For JAX comparisons, reuse
`height_compressed_grad` inside your own loops and adapt `benchmarks/utils.py` to
measure wall time/memory. Contributions that add first-class JAX benchmarks are
very welcome.

