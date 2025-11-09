# API Reference

This page lists the primary symbols intended for external consumption. The project
does not yet claim stable semantic versioning, but the APIs below are the most
useful jumping-off points for experiments and integrations.

## Graph & Analysis

### `tiny_backprop.graph.builders`

- `from_pytorch_fx(fx_graph_module, example_inputs=None)` → `Graph`
- `from_jaxpr(jaxpr)` → `Graph`

Both builders populate activation sizes (when shape-prop information is available)
and run structural validation before returning.

### `tiny_backprop.analysis`

- `compute_frontier_width(graph, order=None)` → `int`
- `frontier_profile(graph, order=None)` → `List[FrontierSnapshot]`
- `summarize_frontier(graph, order=None)` → `FrontierSummary`
- `evaluate_hcp(graph)` → `HCPEvaluation`
- `evaluate_orders(graph)` → `List[OrderStats]`
- `best_heuristic_order(graph)` → `OrderStats`
- `lower_bounds.lower_bound_report(graph)` → `dict[str, int]`
- `analyze_graph(graph, order=None, include_snapshots=True)` → `GraphAnalysisReport`

Use these helpers inside notebooks or offline pipelines to reason about DAG
structure before running the planner.

## Scheduling

- `partition_into_blocks(order, block_size)` → `List[Block]`
- `build_balanced_interval_tree(blocks)` → `IntervalNode`
- `make_interval_plan(graph, block_size)` → `CheckpointPlan`
- `build_replay_plan(checkpoint_plan)` → `ReplayPlan`
- `validate_plan(checkpoint_plan, replay_plan)` → `None`

`CheckpointPlan` exposes `save_nodes`, `interval_tree`, and a metadata dictionary
containing the chosen traversal order, block size, and frontier diagnostics.

## Runtime

- `HeightCompressedExecutor()` — configure via `executor.configure(graph, block_size)`
- `HeightCompressedExecutor.backward(loss, autograd_backward, callbacks=None)`
- `ExecutionCallbacks(run_forward, run_backward, prepare_loss=None, finalize=None)`
- `CheckpointStorage`, `ActivationRegistry`, `Profiler`

When no callbacks are provided, `HeightCompressedExecutor.backward` delegates to the
framework’s `autograd_backward`. With callbacks supplied, the executor iterates over
replay steps (`forward`/`backward`) and lets the integration orchestrate recompute.

## Integrations

### PyTorch

- `HCModule(module: nn.Module, config: Optional[HCConfig] = None)`
- `HCConfig(block_size=None, eager_plan=False, planner=None)`

Example:

```python
from tiny_backprop.integration.torch import HCModule, HCConfig

model = HCModule(my_model, config=HCConfig(block_size=32))
logits = model(inputs)
loss = loss_fn(logits, targets)
model.backward(loss)
```

`HCModule.executor` exposes the most recent `HeightCompressedExecutor` so benchmarks
can inspect `checkpoint_plan.meta`.

### JAX

- `capture_graph_jaxpr(fn, example_inputs)` → `Graph`
- `height_compressed_grad(fn, block_size=None)` → gradient function with attribute
  `last_executor`

Example:

```python
hc_grad = height_compressed_grad(loss_fn, block_size=16)
grads = hc_grad(params)
executor = hc_grad.last_executor
```

If planning fails (e.g. unsupported primitives) the wrapper falls back to plain
`jax.grad` and sets `last_executor` to `None`.

## Benchmarks & Utilities

- `benchmarks.utils.BenchmarkResult`
- `benchmarks.utils.run_single_trial(...)`
- `benchmarks.utils.summarize(results)` → aggregated statistics
- `benchmarks.utils.format_summary_table(summary)` → nice-print table
- `benchmarks.utils.export_json(results, path)` → raw result dump

Benchmarks and experiments rely on these helpers; feel free to reuse them in custom
suites or CI performance regressions.

