# tiny-backprop

**Height-compressed backpropagation for arbitrary computation graphs.**

`tiny-backprop` turns the classic √T-memory checkpointing result for “skinny” computations into a practical toolbox for modern deep networks. The repository is intentionally organised like a research project: you will find the graph IR and analytical tools, the scheduling engine, framework bindings, benchmarks, and extensive documentation in one place.

---

## Table of contents

1. [Features](#features)
2. [Quick start](#quick-start)
3. [Running the full verification suite](#running-the-full-verification-suite)
4. [Benchmarks & workload profiles](#benchmarks--workload-profiles)
5. [Project layout](#project-layout)
6. [Roadmap](#roadmap)
7. [Documentation](#documentation)
8. [Contributing](#contributing)
9. [License](#license)

---

## Features

- **Framework-agnostic IR** – capture DAGs from PyTorch FX or JAXPR and reason about activation sizes, live frontiers, and theoretical lower bounds.
- **Height-compressed scheduler** – partition topological orders into balanced interval trees, plan checkpoints, and build replay schedules that respect chosen memory budgets.
- **Runtime executor** – orchestrate planned recomputation and backward passes while tracking memory, recompute FLOPs, and timing statistics.
- **Drop-in integrations** – `HCModule` for PyTorch and `height_compressed_grad` scaffolding for JAX.
- **Benchmarks & experiments** – turnkey scripts covering Transformers, GPT-style decoders, long-context sweeps, ResNets, U-Nets, and synthetic DAGs.
- **Verification harness** – a single command to lint, test, run smoke checks, execute benchmarks, and summarise efficiency metrics (including CPU-only runs).

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

1. Harden PyTorch FX capture and extend coverage to ONNX / TensorFlow graphs.
2. Implement production-grade frontier minimisation and checkpoint planners beyond the balanced interval tree baseline.
3. Integrate the executor with real training loops (PyTorch Lightning / JAX pjit).
4. Publish reproducible benchmark baselines comparing tiny-backprop, naive autograd, PyTorch checkpointing, and third-party schedulers.
5. Prove and document near-optimality guarantees for broader DAG families (beyond “height-compressible” regimes).

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

- Additional graph builders (ONNX, TensorFlow, etc.).
- Better topological-order heuristics and scheduling strategies.
- Runtime optimisations and integrations with popular training frameworks.
- Expanded benchmark coverage and reproducible baseline reports.
- Documentation improvements and tutorial notebooks.

Please run `./run_full_test --benchmark-profile small --trials 1` before opening a PR.

---

## License

This project is released under the [MIT License](LICENSE).
