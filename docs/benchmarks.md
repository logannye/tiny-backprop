# Benchmark Reproducibility Guide

This document explains how to reproduce the benchmark results quoted in the
tiny-backprop documentation and evaluate height-compressed backpropagation against
baseline strategies.

## 1. Prerequisites

- Python 3.10+
- GPU with CUDA (recommended) or CPU-only (benchmarks will run but absolute numbers
  differ)
- Dev environment prepared via `requirements-dev.txt` (see `docs/testing.md`)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

For CUDA builds of PyTorch/JAX, follow the respective project instructions and
install the wheels before running the benchmarks.

## 2. Baseline Command Matrix

Benchmark CLIs support three execution modes:

- `naive` — vanilla autograd (no checkpointing).
- `checkpoint` — PyTorch checkpointing (layer-wise `torch.utils.checkpoint`) where
  implemented.
- `tiny` — height-compressed executor via tiny-backprop.

Each CLI accepts `--modes`, `--trials`, `--profile`, and `--export` flags. The
Makefile bundles the standard runs:

```bash
make bench-transformer   # transformer (encoder) benchmark
make bench-resnet        # ResNet or CNN fallback
make bench-gpt2          # GPT-style decoder benchmark
make bench-long-context  # context-length sweep
make bench-unet          # diffusion U-Net benchmark
```

Use `--profile {small,medium,large}` to scale workloads. `medium` matches the
defaults shown here, `small` trims dimensions for quick smoke tests, and `large`
pushes models to stress memory.

Benchmarks store raw JSON in the `results/` directory. Adjust `--trials` for
statistical confidence (default is a lightweight 2–3 runs to keep CI fast).

## 3. Interpreting Results

JSON entries conforms to `BenchmarkResult`:

```json
{
  "benchmark": "transformer",
  "mode": "tiny",
  "trial": 0,
  "wall_time_s": 0.842,
  "peak_mem_bytes": 123456789,
  "peak_mem_reserved_bytes": 150000000,
  "peak_cpu_bytes": 524288000,
  "gpu_utilization_pct": 64.5,
  "loss_value": 0.756,
  "extra_metrics": {}
}
```

Use helper utilities for summaries:

```python
from benchmarks.utils import BenchmarkResult, summarize, format_summary_table
import json

with open("results/transformer.json") as fh:
    raw = json.load(fh)
results = [BenchmarkResult(**entry) for entry in raw]
summary = summarize(results)
print(format_summary_table(summary))
```

Example output:

```
mode       | trials | wall_time_mean_s | peak_mem_mean_mb | peak_reserved_mean_mb | peak_cpu_mean_mb | gpu_util_mean_pct | loss_mean
+-----------+--------+------------------+------------------+-----------------------+------------------+-------------------+----------
naive      | 3      | 0.8150           | 950.1234         | 1100.5678             | 512.0000         | 45.0000           | 0.7543
checkpoint | 3      | 0.9200           | 600.0000         | 700.0000              | 410.0000         | 47.5000           | 0.7540
tiny       | 3      | 0.8700           | 400.0000         | 500.0000              | 298.0000         | 50.2000           | 0.7541
```

Interpretation tips:

- **Peak memory**: emphasise reduction vs naive baseline. On GPU runs, compare
  `peak_mem_mean_mb` / `peak_reserved_mean_mb`; on CPU-only hardware, use
  `peak_cpu_mean_mb` to reason about resident RSS.
- **Wall time**: account for recompute overhead; HCB aims to stay within a modest
  slowdown vs naive.
- **GPU utilisation**: best-effort metric from NVML. If `N/A`, install `pynvml`.
  CPU-only runs will typically report utilisation as `None` together with populated
  `peak_cpu_mean_mb` values.
- **Loss**: should match across modes (sanity check).

## 4. Sharing & Regression Tracking

- CI automatically runs a subset of benchmarks on pushes to `main` and uploads
  `results/*.json` as build artifacts (see `.github/workflows/ci.yml`).
- For deeper analysis, load JSON files into notebooks (see `notebooks/benchmark_report.ipynb`)
  to plot memory vs time scatter charts or sequence-length scaling curves.

## 5. Troubleshooting

- **CUDA out-of-memory**: reduce `--batch-size`/`--seq-len`, or run CPU-only baseline.
- **Benchmark variance**: increase `--trials` and pin `TINY_BACKPROP_SEED` for
  determinism.
- **Missing GPU utilisation**: install `pynvml` (`pip install pynvml`) and ensure
  NVIDIA drivers expose NVML; otherwise metrics default to `None`.

## 6. Reporting Results

When presenting efficiency claims:

1. Run each benchmark with sufficient trials (≥5 recommended).
2. Report mean ± standard deviation for wall time and peak memory.
3. Include hardware details (GPU model, driver version, CUDA/JAX versions).
4. Provide raw JSON outputs alongside plots to allow reproduction.

Use this template to summarise key findings in reports or READMEs.

