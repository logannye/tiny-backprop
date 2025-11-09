# tiny-backprop

**Height-compressed backpropagation for arbitrary computation graphs.**

`tiny-backprop` turns the classic √T-memory checkpointing result for “skinny” computations into a practical toolbox for modern deep networks. The repository is intentionally organised like a research project: you will find the graph IR and analytical tools, the scheduling engine, framework bindings, benchmarks, and extensive documentation in one place.

---

## Table of contents

1. [Features](#features)
2. [Design intent vs implementation](#design-intent-vs-implementation)
3. [Theory at a glance](#theory-at-a-glance)
4. [Quick start](#quick-start)
5. [Running the full verification suite](#running-the-full-verification-suite)
6. [Benchmarks & workload profiles](#benchmarks--workload-profiles)
7. [Project layout](#project-layout)
8. [Roadmap](#roadmap)
9. [Documentation](#documentation)
10. [Contributing](#contributing)
11. [License](#license)

---

## Features

- **Framework-agnostic IR** – capture DAGs from PyTorch FX or JAXPR and reason about activation sizes, live frontiers, and theoretical lower bounds.
- **Height-compressed scheduler** – partition topological orders into balanced interval trees, plan checkpoints, and build replay schedules that respect chosen memory budgets.
- **Runtime executor** – orchestrate planned recomputation and backward passes while tracking memory, recompute FLOPs, and timing statistics.
- **Drop-in integrations** – `HCModule` for PyTorch and `height_compressed_grad` scaffolding for JAX.
- **Benchmarks & experiments** – turnkey scripts covering Transformers, GPT-style decoders, long-context sweeps, ResNets, U-Nets, and synthetic DAGs.
- **Verification harness** – a single command to lint, test, run smoke checks, execute benchmarks, and summarise efficiency metrics (including CPU-only runs).

---

## Design intent vs implementation

| Height-compressed backprop objective | Implemented artefacts | Remaining gaps |
| --- | --- | --- |
| Capture computation graphs and expose frontier statistics | `tiny_backprop.graph.*`, `analysis/frontier_width.py`, `analysis/report.py` with unit tests | Importers for ONNX/XLA/TF, richer visual diagnostics |
| Build balanced interval-tree decompositions and checkpoint plans | `schedule/*` (blocks, interval trees, checkpoint & replay planners) plus validators | Adaptive block sizing, cost-aware planners beyond balanced trees |
| Execute height-compressed schedules at runtime | `runtime/executor.py`, `runtime/hooks.py`, `runtime/profiling.py`, smoke tests | GPU scaling studies, heterogeneous memory tiers, preemption-aware execution |
| Provide drop-in framework bindings | `integration/torch.HCModule`, `integration/jax.height_compressed_grad`, worked examples | Production-facing wrappers (Lightning, 🤗 Accelerate) and TensorFlow/ONNX backends |
| Demonstrate efficiency vs baselines | Benchmarks under `benchmarks/` & `experiments/`, consolidated verification summaries | Large-scale runs, automated regression dashboards, third-party baseline comparisons |
| Ground in theory and proofs | `docs/theory.md`, `docs/design_notes.md`, unit tests around frontier width and schedules | Formal general-DAG optimality proofs, explicit lower-bound notebooks |

Taken together, the repository already realises the end-to-end pipeline outlined in the design brief. The next breakthroughs live in generalising the theory, tightening schedules, and broadening framework support.

---

## Theory at a glance

- **Naive baseline**: Reverse-mode autodiff stores every forward activation → peak memory `O(L · w)` where `L` is depth in a chosen order and `w` is frontier width (total live activation size).
- **Height-compressed schedule**: For graphs that satisfy the locality conditions in `docs/theory.md`, the balanced interval-tree planner retains only `O(√L)` block boundaries of width `O(w)`, yielding `O(w · √L)` peak activation memory with low constant-factor recomputation overhead.
- **Supporting references**:
  - `docs/theory.md` – formalises frontier width, relates it to pebbling bounds, and mirrors the Williams-style height compression argument.
  - `docs/design_notes.md` – explains how modules in `src/tiny_backprop` implement the Σ([i,j]) abstraction, schedule validation, and replay invariants.
  - `docs/benchmarks.md` – documents how empirical benchmarks compare `tiny`, `naive`, and `checkpoint` modes and interpret memory/time savings.

These resources show how the implementation operationalises the Williams/Chen/Revolve lineage for general DAGs.

---

## Quick start

```bash
git clone https://github.com/logannye/tiny-backprop.git
cd tiny-backprop

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Run a fast smoke of the verification workflow:

```bash
./run_full_test --benchmark-profile small --trials 1
```

This wrapper auto-discovers `python`/`python3`, forwards all flags to `scripts/run_full_test.py`, and records results under `results/verification_summary_*.json`.

---

## Running the full verification suite

```bash
./run_full_test \
  --benchmark-profile medium \
  --trials 3 \
  --targets transformer gpt2 resnet long_context unet
```

What you get:

- **Lint** (`ruff`), **unit & integration pytest**, and **framework smoke tests**.
- Benchmarks in the requested modes with JSON outputs written to `results/`.
- A consolidated summary (`verification_summary_*.json`) capturing wall time, GPU memory, and CPU resident memory (`peak_cpu_mean_mb`) so CPU-only hardware is fully supported.

Handy flags:

| Flag | Purpose |
| --- | --- |
| `--benchmark-profile {small,medium,large}` | Choose workload size presets. |
| `--targets ...` | Run a subset of benchmarks. |
| `--memory-threshold` / `--time-threshold` | Adjust pass/fail criteria for efficiency checking. |
| `--skip-*` | Skip individual pipeline stages when iterating locally. |

---

## Benchmarks & workload profiles

Every benchmark/experiment script now supports `--profile {small,medium,large}`. Profiles scale sequence lengths, depths, batch sizes, and block sizes so you can move from a fast CI sanity check (`small`) to memory stress tests (`large`) without hand-editing code. Examples:

```bash
# Transformer encoder benchmark (naive vs checkpoint vs tiny)
python -m benchmarks.mem_vs_time_transformer --profile large --trials 5

# GPT-style decoder benchmark
python experiments/transformers/gpt2_mem_bench.py --profile medium

# Diffusion U-Net benchmark
python experiments/diffusion/unet_mem_bench.py --profile large --trials 3
```

Summaries now include:

- `peak_mem_mean_mb` / `peak_reserved_mean_mb` (GPU metrics),
- `peak_cpu_mean_mb` (RSS-based CPU metric, via `psutil`),
- `gpu_util_mean_pct` where NVML is available.

A lightweight `Makefile` exposes shortcuts (`make bench-transformer`, `make bench-resnet`, …) using the default `medium` profile.

**Interpreting the results**:

- Each workload runs in `naive`, `checkpoint`, and `tiny` modes so you can evaluate memory savings and recompute overhead directly.
- `scripts/run_full_test.py` fuses benchmark outputs into `verification_summary_*.json` with derived metrics (`memory_saving`, `time_overhead`) for quick comparison.
- `docs/benchmarks.md` provides guidance on crafting publishable claims (e.g., “tiny-backprop delivers 3.4× lower activation memory than naive autograd on GPT-2 medium with ≤1.2× runtime overhead”).
- Raw JSON exports in `results/` preserve all telemetry for audits, plotting, or regression analysis.

---

## Project layout

```
benchmarks/            Benchmark CLIs + utilities
docs/                  In-depth design docs, testing & benchmark guides, theory notes
experiments/           Research-grade experiments (Transformers, U-Nets, synthetic DAGs, ...)
scripts/run_full_test.py
                       End-to-end verification harness
src/tiny_backprop/     Library source (graph IR, analysis, scheduling, runtime, integrations)
tests/                 Unit and integration test suites
results/               Placeholder directory for exported benchmark artefacts
```

See `docs/testing.md` and `docs/benchmarks.md` for detailed instructions.

---

## Roadmap

1. **Graph capture & visualisation** – Broaden importer support (ONNX, TensorFlow, XLA) and ship frontier-width inspection tooling for debugging schedules.
2. **Scheduling optimisation** – Explore adaptive interval trees, heterogeneous memory tiers, and cost-aware planners that minimise recomputation under strict memory caps.
3. **Runtime integration** – Harden GPU execution, support distributed stacks (Lightning, FSDP, JAX `pjit`), and add observability hooks for production use.
4. **Benchmark automation** – Maintain CI-grade suites at small/medium/large profiles, track regressions against naive and checkpoint baselines, and surface dashboards.
5. **Theory formalisation** – Publish general-DAG optimality proofs, codify lower bounds, and package proof artefacts/notebooks alongside code for review.

Contributions against this roadmap are very welcome—see below.

---

## Documentation

- [docs/testing.md](docs/testing.md): environment setup, pytest markers, verification script usage.
- [docs/benchmarks.md](docs/benchmarks.md): benchmark methodology, interpreting metrics, reporting guidelines.
- [docs/theory.md](docs/theory.md): mathematical background on height-compressed backpropagation.
- [docs/integration_guide_torch.md](docs/integration_guide_torch.md) / [docs/integration_guide_jax.md](docs/integration_guide_jax.md): framework-specific integration notes.
- [docs/design_notes.md](docs/design_notes.md): implementation notes and architectural decisions.

---

## Contributing

Issues and PRs are open. Useful contributions include:

- **Graph ingestion** – ONNX, TensorFlow, XLA, or Triton capture paths plus graph visualisers to inspect frontier width and schedule layouts.
- **Scheduling & theory** – New ordering heuristics, adaptive interval trees, heterogenous-memory planners, and proof polish for the guarantees described in `docs/theory.md`.
- **Runtime & integrations** – Integrations with Lightning, 🤗 Accelerate, DeepSpeed, or JAX distributed primitives; validation for AMP/mixed-precision and CPU offload.
- **Benchmarks & analysis** – Larger-scale runs, regression dashboards, additional baselines (framework checkpoint APIs, third-party rematerialisation tools).
- **Documentation & tutorials** – End-to-end notebooks, practitioner guides anchored in the Williams-style height compression narrative, and translation of proofs into accessible walkthroughs.

Please run `./run_full_test --benchmark-profile small --trials 1` before opening a PR.

---

## License

This project is released under the [MIT License](LICENSE).
