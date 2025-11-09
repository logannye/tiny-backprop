# tiny-backprop Documentation

Welcome to the developer documentation for **tiny-backprop**. This site explains the
core ideas behind height-compressed backpropagation, how the implementation is
structured, and how to integrate the planner/runtime with PyTorch and JAX workloads.

## Contents

- [Project Overview](#project-overview)
- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Benchmark & Experiment Suite](#benchmark--experiment-suite)
- [Further Reading](#further-reading)

## Project Overview

`tiny-backprop` turns height-compressed backpropagation (HCB) into a practical system:

1. **Graph capture** normalises framework traces (PyTorch FX, JAXPR) into a DAG.
2. **Analysis** estimates live frontier width, checks height-compressibility, and
   produces lower bounds to aim for.
3. **Scheduling** partitions the order into blocks, builds an interval tree, and
   selects checkpoints that bound peak activation memory.
4. **Runtime** executes the replay plan, offering hooks for storage, profiling, and
   integration callbacks.
5. **Experiments & benchmarks** compare naive autograd, conventional checkpointing,
   and height-compressed execution on real models.

The implementation emphasises modularity so researchers can tweak heuristics or swap
in new planning strategies while practitioners leverage the out-of-the-box wrappers.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run formatter/lints (if configured)
ruff check .

# Execute unit and integration tests (requires pytest, torch, jax, torchvision)
pytest

# Launch a benchmark (see --help for options)
python -m benchmarks.mem_vs_time_transformer --trials 3 --export results/transformer.json
```

For GPU-focused experiments ensure CUDA drivers and libraries are available.

## Repository Layout

| Directory | Purpose |
|-----------|---------|
| `src/tiny_backprop/graph` | Graph IR, builders for PyTorch FX/JAXPR, topo utilities. |
| `src/tiny_backprop/analysis` | Frontier width, heuristics, height-compressibility diagnostics, reporting. |
| `src/tiny_backprop/schedule` | Block partitioning, interval tree construction, checkpoint/replay planning, validation. |
| `src/tiny_backprop/runtime` | Execution engine, checkpoint storage, activation registry, profiling helpers. |
| `src/tiny_backprop/integration` | Framework adapters (`torch`, `jax`) and examples. |
| `benchmarks/` | CLI benchmarks for transformers, ResNets, U-Nets with reporting utilities. |
| `experiments/` | Research scripts sweeping context length, depth, synthetic DAGs, etc. |
| `docs/` | Documentation (you are here). |
| `tests/` | Unit and integration tests covering each subsystem. |

## Benchmark & Experiment Suite

- **Benchmarks** (`benchmarks/`): provide reproducible comparisons between naive,
  checkpoint, and height-compressed training. The utilities measure wall-clock time,
  peak memory, and loss values. Results can be exported to JSON for dashboards.
- **Experiments** (`experiments/`): focus on research diagnostics—context-length
  sweeps, synthetic DAG analysis, diffusion structures. They reuse the same runtime
  primitives but aim to answer structural questions (How skinny is my graph? What
  block size hits the √L memory regime?).

Use the benchmarks for regression and performance tracking, and experiments for
research exploration or paper figures.

## Further Reading

- [Theory](theory.md) — mathematical background for height-compressed backprop.
- [Design Notes](design_notes.md) — architectural decisions and component APIs.
- [API Reference](api_reference.md) — key public-facing classes/functions.
- [PyTorch Integration Guide](integration_guide_torch.md) — end-to-end usage.
- [JAX Integration Guide](integration_guide_jax.md) — early-stage JAX support.

