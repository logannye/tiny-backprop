# Testing Guide

This document describes how to set up a repeatable environment for running
tiny-backprop’s linters, unit/integration tests, and benchmarks.

## 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
```

> ℹ️ PyTorch and JAX binaries are platform-specific. The default requirements file
> installs CPU wheels. If you require CUDA builds, refer to the upstream
> installation instructions and adjust the extras accordingly.

### Core Dependencies

| Library         | Purpose                                   |
|-----------------|-------------------------------------------|
| `torch`, `torchvision` | PyTorch runtime and reference models      |
| `jax`, `jaxlib` | JAX integration and gradient tests         |
| `pytest`        | Unit/integration test runner              |
| `ruff`          | Linting / formatting enforcement          |
| `numpy`, `scipy`| Shared math utilities for tests/benchmarks|
| `psutil`        | CPU memory sampling for benchmarks        |

## 2. Determinism & Environment Variables

Set the following variables to obtain reproducible results during test runs:

```bash
export PYTHONHASHSEED=0
export TINY_BACKPROP_SEED=1234
export CUDA_LAUNCH_BLOCKING=1        # (optional) helps debugging CUDA kernels
```

PyTorch/JAX seeds are initialised in the test fixtures, but you can override them by
exporting `TINY_BACKPROP_SEED`.

When running on GPU, ensure drivers and CUDA runtime match the torch/jax wheels.

## 3. Test Commands

The project’s `Makefile` exposes convenient shortcuts (run from repo root):

```bash
make lint          # ruff linting / formatting check
make test          # fast unit tests
make test-full     # full pytest suite, including integration tests
```

### Direct `pytest` Usage

```bash
# Unit tests only
pytest tests/unit

# Integration tests (PyTorch + JAX)
pytest tests/integration

# Run slow tests explicitly
pytest -m slow
```

Pytest markers (`slow`, `gpu`) are defined in `pytest.ini`. GPU-tagged tests are
skipped automatically if CUDA is unavailable.

## 4. Benchmark Commands

Benchmarks live under `benchmarks/` and `experiments/`. Use the provided CLIs:

```bash
make bench-transformer      # transformer memory/time benchmark
make bench-resnet           # resnet benchmark
make bench-gpt2             # GPT-style benchmark
make bench-long-context     # sequence-length sweep
make bench-unet             # diffusion U-Net benchmark
```

Each command now exposes workload profiles in addition to the usual flags:

```bash
python -m benchmarks.mem_vs_time_transformer --profile medium
```

Profiles (`small|medium|large`) scale batch size, sequence/image dimensions, and
model depth so you can choose between quick smoke runs and stress tests. All
profiled CLIs still accept explicit overrides (`--batch-size`, `--seq-len`, ...)
if you need custom settings.

Benchmark runs produce JSON files in the `results/` directory; combine them with the
report notebook under `notebooks/` to visualise outcomes. Consult `docs/benchmarks.md`
for interpretation guidance and reproducibility details.

## 6. Full Verification Script

Run everything (lint, pytest, smoke tests, and benchmarks) via:

```bash
python scripts/run_full_test.py --benchmark-profile medium
```

Useful flags:

- `--benchmark-profile {small,medium,large}`: pass workload profiles through to
  every benchmark CLI.
- `--targets ...`: run a subset of benchmarks (e.g. `--targets transformer resnet`).
- `--memory-threshold` / `--time-threshold`: adjust pass/fail criteria for the
  efficiency summary.

The script emits per-benchmark JSON files plus a consolidated
`results/verification_summary_*.json` containing wall time, GPU memory, and
CPU-resident memory (`peak_cpu_mean_mb`), making it easier to justify efficiency
claims on CPU-only hardware.

## 5. Troubleshooting

- **CUDA out-of-memory**: reduce `--batch-size`, `--seq-len`, or switch modes to
  `naive` to confirm baseline fits.
- **Missing GPU**: GPU-specific tests/benchmarks are automatically skipped; rerun
  once CUDA hardware is available.
- **FX tracing errors**: some PyTorch modules rely on control flow or script-only
  ops. Wrap submodules manually or provide a custom planner via `HCConfig`.

For additional guidance, check `docs/benchmarks.md` (benchmark methodology) and
open issues/PRs to discuss platform-specific challenges.

