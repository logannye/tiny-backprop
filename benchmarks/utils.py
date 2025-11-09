from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence

import torch

try:  # optional GPU utilization
    import pynvml  # type: ignore

    _NVML_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore
    _NVML_AVAILABLE = False

try:  # optional CPU memory tracking
    import psutil  # type: ignore

    _PSUTIL_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore
    _PSUTIL_AVAILABLE = False

_NVML_INITIALISED = False


def _maybe_init_nvml() -> None:
    global _NVML_INITIALISED  # noqa: PLW0603
    if _NVML_AVAILABLE and not _NVML_INITIALISED:
        try:
            pynvml.nvmlInit()
            _NVML_INITIALISED = True
        except Exception:  # pragma: no cover - NVML issues are non-fatal
            pass


@dataclass
class BenchmarkResult:
    """
    Structured summary for a single benchmark trial.
    """

    benchmark: str
    mode: str
    trial: int
    wall_time_s: float
    peak_mem_bytes: int
    peak_mem_reserved_bytes: int
    gpu_utilization_pct: Optional[float]
    loss_value: float
    peak_cpu_bytes: Optional[int] = None
    extra_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = asdict(self)
        return result


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return device.index
    return torch.cuda.current_device()


def _get_gpu_utilisation(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    _maybe_init_nvml()
    if not _NVML_INITIALISED:
        return None
    try:
        idx = _device_index(device)
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:  # pragma: no cover - utilisation best-effort
        return None


def run_single_trial(
    benchmark_name: str,
    mode: str,
    trial: int,
    *,
    device: torch.device,
    forward_fn: Callable[[], torch.Tensor],
    backward_fn: Callable[[torch.Tensor], None],
    extra_metrics: Optional[Dict[str, float]] = None,
) -> BenchmarkResult:
    """
    Execute a single forward/backward pass and record time/memory stats.

    Args:
        benchmark_name: Label for the benchmark family (e.g. "transformer").
        mode: Execution mode ("naive", "tiny", etc.).
        trial: Integer trial index.
        device: torch.device used for the run.
        forward_fn: Callable returning the scalar loss tensor.
        backward_fn: Callable that consumes the loss tensor and performs backward.
        extra_metrics: Optional dictionary of custom metrics to attach.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    proc = psutil.Process() if _PSUTIL_AVAILABLE else None
    rss_before = proc.memory_info().rss if proc else None

    start = perf_counter()
    loss = forward_fn()
    _sync_device(device)
    backward_fn(loss)
    _sync_device(device)
    wall = perf_counter() - start

    peak_allocated = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    )
    gpu_util = _get_gpu_utilisation(device)
    peak_cpu = None
    if proc and rss_before is not None:
        rss_after = proc.memory_info().rss
        peak_cpu = max(rss_before, rss_after)

    return BenchmarkResult(
        benchmark=benchmark_name,
        mode=mode,
        trial=trial,
        wall_time_s=wall,
        peak_mem_bytes=int(peak_allocated),
        peak_mem_reserved_bytes=int(peak_reserved),
        peak_cpu_bytes=int(peak_cpu) if peak_cpu is not None else None,
        gpu_utilization_pct=gpu_util,
        loss_value=float(loss.detach().item()),
        extra_metrics=extra_metrics or {},
    )


def summarize(results: Sequence[BenchmarkResult]) -> List[dict]:
    """
    Aggregate benchmark results by mode and return summaries suitable for printing.
    """
    summaries: List[dict] = []
    by_mode: dict[str, List[BenchmarkResult]] = {}
    for res in results:
        by_mode.setdefault(res.mode, []).append(res)

    for mode, group in sorted(by_mode.items(), key=lambda kv: kv[0]):
        gpu_utils = [g for g in (r.gpu_utilization_pct for r in group) if g is not None]
        summaries.append(
            {
                "mode": mode,
                "trials": len(group),
                "wall_time_mean_s": mean(r.wall_time_s for r in group),
                "peak_mem_mean_mb": mean(r.peak_mem_bytes for r in group) / 1e6,
                "peak_reserved_mean_mb": mean(r.peak_mem_reserved_bytes for r in group)
                / 1e6,
                "peak_cpu_mean_mb": (
                    mean(r.peak_cpu_bytes for r in group if r.peak_cpu_bytes is not None) / 1e6
                )
                if any(r.peak_cpu_bytes is not None for r in group)
                else None,
                "gpu_util_mean_pct": mean(gpu_utils) if gpu_utils else None,
                "loss_mean": mean(r.loss_value for r in group),
            }
        )
    return summaries


def export_json(results: Sequence[BenchmarkResult], destination: Path) -> None:
    """
    Write raw benchmark results to JSON for later analysis.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [res.to_dict() for res in results]
    destination.write_text(json.dumps(payload, indent=2))


def _format_value(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def format_summary_table(summary: Sequence[dict]) -> str:
    """
    Format aggregated summaries into a readable table.
    """
    if not summary:
        return "No results recorded."

    headers = [
        "mode",
        "trials",
        "wall_time_mean_s",
        "peak_mem_mean_mb",
        "peak_reserved_mean_mb",
        "peak_cpu_mean_mb",
        "gpu_util_mean_pct",
        "loss_mean",
    ]

    col_widths: dict[str, int] = {}
    for header in headers:
        max_len = len(header)
        for row in summary:
            value = row[header]
            cell = (
                _format_value(value)
                if isinstance(value, (float, type(None)))
                else str(value)
            )
            max_len = max(max_len, len(cell))
        col_widths[header] = max_len

    def format_cell(h: str, value: object) -> str:
        if isinstance(value, (float, type(None))):
            return _format_value(value).ljust(col_widths[h])
        return str(value).ljust(col_widths[h])

    lines = [
        " | ".join(h.ljust(col_widths[h]) for h in headers),
        "-+-".join("-" * col_widths[h] for h in headers),
    ]
    for row in summary:
        lines.append(" | ".join(format_cell(h, row[h]) for h in headers))
    return "\n".join(lines)

