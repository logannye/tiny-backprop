# Design Notes

This document captures the architectural decisions, data flows, and extension
points within `tiny-backprop`. It is aimed at contributors who wish to evolve the
scheduler, plug in new analyses, or integrate alternative runtimes.

## Guiding Principles

1. **Graph-first worldview**: every framework reduces to a DAG where nodes carry
   metadata (activation size, operator type), enabling tool-agnostic scheduling.
2. **Layered responsibilities**: graph capture → analysis → scheduling → runtime →
   integrations. Each layer depends only on the one beneath it.
3. **Provable levers**: analysis retains enough signal (frontier width, cut bounds)
   to reason about optimality gaps. Scheduling strategies can introspect those
   diagnostics before execution.
4. **Drop-in ergonomics**: PyTorch and JAX adapters aim for a single-line wrapper
   while still exposing knobs (block size, planner pipelines).

## Component Walkthrough

### Graph Layer (`src/tiny_backprop/graph`)

- `builders.py`: translates PyTorch FX `GraphModule` and JAX `ClosedJaxpr` into a
  shared `Graph`/`Node` IR. Captures activation size estimates (tensor meta) to
  inform frontier calculations.
- `ir.py`: lightweight DAG container supporting validation, topological sort,
  induced subgraphs, and frontier profiling.
- `topo.py` & `visualize.py`: house default topological order selection and DOT
  export helpers for debugging.

**Extension tips**: add new builders (e.g. ONNX, MLIR) by producing `Node` instances
with well-defined activation sizes. Ensure `Graph.validate()` succeeds to catch
cycles or missing inputs early.

### Analysis Layer (`src/tiny_backprop/analysis`)

- `frontier_width.py`: produces detailed frontier snapshots and summaries
  (`FrontierSummary`) that record peak, tail, and average live memory.
- `width_heuristics.py`: hosts multiple topological order heuristics. The default
  greedy strategy optimises for local live-set minimisation.
- `hcp_criteria.py`: heuristically evaluates whether a graph is height-compressible,
  combining frontier ratios, depth, and fan-out metrics.
- `lower_bounds.py`: aggregates simple lower bounds (max activation, heuristic
  frontier) for near-optimality comparisons.
- `report.py`: single entry point to analyse a graph and emit structured diagnostics.

**Extension tips**: when adding heuristics, return `OrderStats` entries from
`evaluate_orders` and the best order will propagate to checkpoint planning.

### Scheduler Layer (`src/tiny_backprop/schedule`)

- `block_partition.py`: partitions an ordered node list into contiguous blocks.
- `interval_tree.py`: builds a balanced interval tree over those blocks to guide
  recursive replay.
- `checkpoint_plan.py`: selects block boundaries as checkpoints, computes frontier
  metadata, and packages the plan.
- `replay_plan.py`: converts the interval tree into forward/backward replay steps
  (post-order traversal to ensure dependencies are satisfied).
- `validate.py`: ensures checkpoint indices are sorted, interval trees are
  consistent, and replay steps align with the tree.

**Extension tips**: replace `_select_checkpoints` with smarter policies (e.g.
  dynamic programming on tree nodes) while reusing validation and replay plumbing.

### Runtime Layer (`src/tiny_backprop/runtime`)

- `executor.py`: orchestrates execution. Given a graph, it builds a plan, validates
  it, and executes replay steps via user-provided callbacks (`ExecutionCallbacks`).
  If no callbacks are provided, it gracefully falls back to native autograd.
- `hooks.py` & `storage.py`: simple activation/ checkpoint registries ready to be
  replaced with device-aware implementations (e.g. CPU offload, NVMe).
- `profiling.py`: collects peak memory, recompute FLOPs, and event durations.

**Extension tips**: adapt `CheckpointStorage` to support hybrid host/device tiers by
overriding `put/get/delete`. Collect fine-grained metrics by decorating callbacks
with profiler hooks.

### Integrations (`src/tiny_backprop/integration`)

- **PyTorch**: `HCModule` wraps `nn.Module`, captures FX graphs on-demand, configures
  executors, and preserves gradient parity. Configuration allows custom planners
  or block sizes. Integration tests compare to native autograd.
- **JAX**: `height_compressed_grad` wraps `jax.grad` to plan a schedule alongside
  gradient evaluations. The gradient function exposes `last_executor` for inspection.

**Extension tips**: JAX support is intentionally minimal—extend by wrapping `jit`,
`pmap`, or by capturing dynamic shapes. For frameworks lacking autograd callback
hooks, bridge through the runtime `execute` API.

## Data Flow Summary

1. **Capture** `Graph` via FX or JAXPR.
2. **Analyse** with `analysis.report.analyze_graph` (optional diagnostic stage).
3. **Schedule** using `make_interval_plan` → `build_replay_plan`.
4. **Validate** with `validate_plan`.
5. **Execute** through `HeightCompressedExecutor.backward` using framework-specific
   callbacks.

## Testing Strategy

- Unit tests live under `tests/unit/` and cover each module (graph IR, scheduler,
  runtime utilities, benchmark helpers).
- Integration tests under `tests/integration/` compare gradients, verify plans on
  realistic models, and ensure JAX/PyTorch bindings stay in sync.
- Benchmarks provide a soft regression harness: use `benchmarks/run_all.sh` to catch
  runtime regressions beyond functional correctness.

## Future Directions

- Stronger checkpoint selection via solver-based optimisation.
- CPU/offload-aware storage managers with asynchronous transfer.
- Auto-tuning of block sizes based on frontier analytics.
- Tighter theory-experiment bridge: automatically report gaps vs lower bounds within
  the experiment scripts and benchmarks.

