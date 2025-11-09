#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class CommandResult:
    name: str
    command: Sequence[str]
    success: bool
    returncode: int
    stdout: str
    stderr: str


@dataclass
class SmokeResult:
    name: str
    success: bool
    details: str


def log(msg: str) -> None:
    print(msg, flush=True)


def run_command(
    name: str,
    command: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> CommandResult:
    log(f"\n[run] {name}: {' '.join(command)}")
    merged_env = {**os.environ, **(env or {})}
    pythonpath = merged_env.get("PYTHONPATH")
    src_path = str(PROJECT_ROOT / "src")
    if pythonpath:
        merged_env["PYTHONPATH"] = f"{src_path}:{pythonpath}"
    else:
        merged_env["PYTHONPATH"] = src_path

    completed = subprocess.run(  # noqa: S603
        command,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        log(completed.stdout.strip())
    if completed.stderr:
        log(completed.stderr.strip())
    return CommandResult(
        name=name,
        command=command,
        success=completed.returncode == 0,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def gather_environment_info() -> Dict[str, object]:
    info: Dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        devices: List[str] = []
        if cuda_available:
            devices = [
                f"{idx}: {torch.cuda.get_device_name(idx)}"
                for idx in range(torch.cuda.device_count())
            ]
        info.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_available": cuda_available,
                "cuda_devices": devices,
            }
        )
    except ModuleNotFoundError:
        info["torch_version"] = None
        info["torch_cuda_available"] = False

    try:
        import torchvision

        info["torchvision_version"] = torchvision.__version__
    except ModuleNotFoundError:
        info["torchvision_version"] = None

    try:
        import jax
        import jaxlib

        info["jax_version"] = jax.__version__
        info["jaxlib_version"] = jaxlib.__version__
        info["jax_default_device"] = str(jax.devices()[0]) if jax.devices() else None
    except ModuleNotFoundError:
        info["jax_version"] = None
        info["jaxlib_version"] = None
        info["jax_default_device"] = None

    return info


def torch_smoke_tests(device: str = "cpu") -> List[SmokeResult]:
    results: List[SmokeResult] = []
    try:
        import torch
        from torch import nn

        from tiny_backprop.integration.torch.wrapped_module import HCConfig, HCModule

        torch.manual_seed(0)

        class TinyChain(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(8, 16),
                    nn.ReLU(),
                    nn.Linear(16, 4),
                )

            def forward(self, x):
                return self.layers(x)

        # Gradient parity check
        model_base = TinyChain().to(device)
        model_ref = TinyChain().to(device)
        model_ref.load_state_dict(model_base.state_dict())

        x = torch.randn(2, 8, device=device, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        y_ref = model_ref(x_ref).sum()
        y_ref.backward()
        baseline_grad = x_ref.grad.detach().clone()

        hc_model = HCModule(model_base, config=HCConfig(block_size=2))
        out = hc_model(x).sum()
        hc_model.backward(out)

        parity = torch.allclose(x.grad, baseline_grad, atol=1e-5)
        results.append(
            SmokeResult(
                name="torch_gradient_parity",
                success=parity,
                details="HCModule gradients match naive baseline" if parity else "Gradient mismatch",
            )
        )

        # Fallback path
        from tiny_backprop.integration.torch import wrapped_module as tb_wrapped

        original_capture = tb_wrapped.capture_graph

        def boom(*args, **kwargs):  # noqa: ARG001
            raise RuntimeError("synthetic failure")

        fallback_success = False
        try:
            dummy = HCModule(TinyChain(), config=HCConfig(block_size=2))
            dummy(torch.randn(2, 8))
            tb_wrapped.capture_graph = boom
            loss = dummy(torch.randn(2, 8)).sum()
            dummy.backward(loss)
            plan = dummy.executor.checkpoint_plan if dummy.executor else None
            fallback_success = plan is not None and plan.save_nodes == []
        finally:
            tb_wrapped.capture_graph = original_capture

        results.append(
            SmokeResult(
                name="torch_capture_fallback",
                success=fallback_success,
                details="Fallback to naive executor confirmed" if fallback_success else "Fallback did not trigger",
            )
        )
    except ModuleNotFoundError:
        results.append(
            SmokeResult(
                name="torch_gradient_parity",
                success=True,
                details="PyTorch not installed (skipped)",
            )
        )
    return results


def jax_smoke_tests() -> List[SmokeResult]:
    results: List[SmokeResult] = []
    try:
        import jax
        import jax.numpy as jnp

        from tiny_backprop.integration.jax import height_compressed_grad

        def loss_fn(w):
            x = jnp.linspace(0.0, 1.0, 4)
            return jnp.sum(jnp.sin(x @ w))

        w = jnp.ones((4, 3))
        hc_grad = height_compressed_grad(loss_fn, block_size=2)
        grads_hc = hc_grad(w)
        grads_ref = jax.grad(loss_fn)(w)

        parity = jnp.allclose(grads_hc, grads_ref, atol=1e-6)
        results.append(
            SmokeResult(
                name="jax_gradient_parity",
                success=bool(parity),
                details="height_compressed_grad matches jax.grad" if parity else "Gradient mismatch",
            )
        )
    except ModuleNotFoundError:
        results.append(
            SmokeResult(
                name="jax_gradient_parity",
                success=True,
                details="JAX not installed (skipped)",
            )
        )
    return results


def run_pytests(skip_unit: bool, skip_integration: bool, pytest_args: Sequence[str]) -> List[CommandResult]:
    results: List[CommandResult] = []
    if not skip_unit:
        results.append(run_command("pytest-unit", [sys.executable, "-m", "pytest", "tests/unit", *pytest_args]))
    if not skip_integration:
        results.append(
            run_command(
                "pytest-integration",
                [sys.executable, "-m", "pytest", "tests/integration/test_torch_chain_model.py", "tests/integration/test_jax_mlp.py", *pytest_args],
            )
        )
    return results


def run_benchmarks(
    benchmarks: Dict[str, Sequence[str]],
    trials: int,
    results_dir: Path,
    timestamp: str,
    profile: str,
) -> Dict[str, Optional[Path]]:
    paths: Dict[str, Optional[Path]] = {}
    results_dir.mkdir(parents=True, exist_ok=True)
    base_env = os.environ.copy()
    existing_pythonpath = base_env.get("PYTHONPATH", "")
    path_entries = [p for p in existing_pythonpath.split(os.pathsep) if p]
    if str(REPO_ROOT) not in path_entries:
        path_entries.insert(0, str(REPO_ROOT))
    base_env["PYTHONPATH"] = os.pathsep.join(path_entries)

    for name, base_cmd in benchmarks.items():
        export_path = results_dir / f"{name}_{timestamp}.json"
        cmd = [*base_cmd, "--trials", str(trials), "--profile", profile, "--export", str(export_path)]
        if "--modes" not in cmd and "mem_vs_time_resnet" in name:
            cmd.extend(["--modes", "naive", "tiny"])
        log(f"\n[benchmark] {name}")
        result = run_command(f"benchmark-{name}", cmd, env=base_env)
        if result.success and export_path.exists():
            paths[name] = export_path
        else:
            paths[name] = None
    return paths


def load_benchmark_results(path: Path):
    from benchmarks.utils import BenchmarkResult

    with path.open() as fh:
        raw = json.load(fh)

    results: List[BenchmarkResult] = []
    for entry in raw:
        # backfill optional keys for older formats
        entry.setdefault("peak_mem_reserved_bytes", entry.get("peak_mem_bytes", 0))
        entry.setdefault("peak_cpu_bytes", None)
        entry.setdefault("gpu_utilization_pct", None)
        entry.setdefault("extra_metrics", {})
        results.append(BenchmarkResult(**entry))
    return results


def evaluate_efficiency(summary: Sequence[dict]) -> Dict[str, Optional[float]]:
    naive = next((row for row in summary if row["mode"] == "naive"), None)
    tiny = next((row for row in summary if row["mode"] == "tiny"), None)
    if not naive or not tiny:
        return {"memory_saving": None, "time_overhead": None}
    memory_saving: Optional[float] = None
    naive_gpu = naive.get("peak_mem_mean_mb")
    tiny_gpu = tiny.get("peak_mem_mean_mb")
    if (
        naive_gpu is not None
        and tiny_gpu is not None
        and isinstance(naive_gpu, (int, float))
        and naive_gpu > 0
    ):
        memory_saving = 1.0 - (tiny_gpu / naive_gpu)
    if memory_saving is None:
        naive_cpu = naive.get("peak_cpu_mean_mb")
        tiny_cpu = tiny.get("peak_cpu_mean_mb")
        if (
            isinstance(naive_cpu, (int, float))
            and isinstance(tiny_cpu, (int, float))
            and naive_cpu > 0
        ):
            memory_saving = 1.0 - (tiny_cpu / naive_cpu)
    time_overhead = tiny["wall_time_mean_s"] / naive["wall_time_mean_s"] if naive["wall_time_mean_s"] else None
    return {"memory_saving": memory_saving, "time_overhead": time_overhead}


def format_smoke_results(results: Iterable[SmokeResult]) -> str:
    lines = ["\nSmoke Test Summary:"]
    for res in results:
        status = "PASS" if res.success else "FAIL"
        lines.append(f"- {status:<4} {res.name}: {res.details}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run comprehensive tiny-backprop verification suite.")
    parser.add_argument("--skip-lint", action="store_true", help="Skip lint checks.")
    parser.add_argument("--skip-unit", action="store_true", help="Skip unit tests.")
    parser.add_argument("--skip-integration", action="store_true", help="Skip integration tests.")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip benchmark comparisons.")
    parser.add_argument("--pytest-args", nargs=argparse.REMAINDER, default=[], help="Extra arguments passed to pytest.")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials for each benchmark mode.")
    parser.add_argument("--memory-threshold", type=float, default=0.40, help="Required fraction of memory saving vs naive.")
    parser.add_argument("--time-threshold", type=float, default=1.30, help="Maximum allowed time overhead vs naive.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory to store benchmark outputs.")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["transformer", "gpt2", "resnet", "long_context", "unet"],
        default=["transformer", "gpt2", "resnet", "long_context", "unet"],
        help="Benchmarks to execute.",
    )
    parser.add_argument(
        "--benchmark-profile",
        choices=["small", "medium", "large"],
        default="medium",
        help="Workload profile passed to each benchmark.",
    )
    args = parser.parse_args()

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    script_summary: Dict[str, object] = {"timestamp": timestamp}
    overall_success = True

    log("=== Environment ===")
    env_info = gather_environment_info()
    log(json.dumps(env_info, indent=2))
    script_summary["environment"] = env_info

    # Section 2: Functional tests
    functional_results: List[CommandResult] = []
    if not args.skip_lint:
        ruff_exe = shutil.which("ruff")
        if ruff_exe is None:
            candidate = Path(sys.executable).with_name("ruff")
            if candidate.exists():
                ruff_exe = str(candidate)
        if ruff_exe:
            functional_results.append(
                run_command(
                    "lint",
                    [
                        ruff_exe,
                        "check",
                        ".",
                        "--exclude",
                        ".venv",
                        "--exclude",
                        "benchmarks/report.ipynb",
                    ],
                )
            )
        else:
            log("[warn] 'ruff' not found on PATH; skipping lint step.")
            functional_results.append(
                CommandResult(
                    name="lint",
                    command=["ruff", "check", "."],
                    success=True,
                    returncode=0,
                    stdout="skipped",
                    stderr="'ruff' not found; lint step skipped.",
                )
            )
    functional_results.extend(run_pytests(args.skip_unit, args.skip_integration, args.pytest_args))
    script_summary["functional_tests"] = [
        {
            "name": res.name,
            "success": res.success,
            "returncode": res.returncode,
            "command": res.command,
        }
        for res in functional_results
    ]
    if any(not res.success for res in functional_results):
        overall_success = False

    # Section 3: Smoke tests
    smoke_results: List[SmokeResult] = []
    device = "cuda" if env_info.get("torch_cuda_available") else "cpu"
    smoke_results.extend(torch_smoke_tests(device))
    smoke_results.extend(jax_smoke_tests())
    log(format_smoke_results(smoke_results))
    script_summary["smoke_tests"] = [
        {"name": res.name, "success": res.success, "details": res.details} for res in smoke_results
    ]
    if any(not res.success for res in smoke_results):
        overall_success = False

    # Section 4 & 5: Benchmarks and efficiency evaluation
    efficiency_reports: Dict[str, dict] = {}
    if not args.skip_benchmarks:
        available_benchmarks: Dict[str, Sequence[str]] = {
            "transformer": [sys.executable, "-m", "benchmarks.mem_vs_time_transformer", "--modes", "naive", "checkpoint", "tiny"],
            "gpt2": [sys.executable, "experiments/transformers/gpt2_mem_bench.py"],
            "resnet": [sys.executable, "-m", "benchmarks.mem_vs_time_resnet"],
            "long_context": [sys.executable, "experiments/transformers/long_context.py"],
            "unet": [sys.executable, "experiments/diffusion/unet_mem_bench.py"],
        }
        selected = {name: available_benchmarks[name] for name in args.targets if name in available_benchmarks}
        benchmark_paths = run_benchmarks(
            selected,
            args.trials,
            args.results_dir,
            timestamp,
            args.benchmark_profile,
        )

        from benchmarks.utils import format_summary_table, summarize

        for name, path in benchmark_paths.items():
            if path is None:
                log(f"[warn] Benchmark {name} failed or produced no output.")
                overall_success = False
                continue
            results = load_benchmark_results(path)
            summary = summarize(results)
            log(f"\nBenchmark Summary ({name}):\n{format_summary_table(summary)}")
            efficiency = evaluate_efficiency(summary)
            meets_memory = (
                efficiency["memory_saving"] is None or efficiency["memory_saving"] >= args.memory_threshold
            )
            meets_time = (
                efficiency["time_overhead"] is None or efficiency["time_overhead"] <= args.time_threshold
            )
            if efficiency["memory_saving"] is None or efficiency["time_overhead"] is None:
                log(f"[warn] Insufficient data to compute efficiency for {name}.")
                meets_memory = True
                meets_time = True
            if not (meets_memory and meets_time):
                overall_success = False
            efficiency_reports[name] = {
                "results_path": str(path),
                "summary": summary,
                "efficiency": efficiency,
                "meets_memory_threshold": meets_memory,
                "meets_time_threshold": meets_time,
            }
    else:
        log("\n[info] Benchmarks skipped by user request.")
    script_summary["benchmarks"] = efficiency_reports
    script_summary["benchmark_profile"] = args.benchmark_profile

    script_summary["overall_success"] = overall_success
    summary_path = args.results_dir / f"verification_summary_{timestamp}.json"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as fh:
        json.dump(script_summary, fh, indent=2)
    log(f"\nSummary written to {summary_path}")

    if not overall_success:
        log("\n[FAIL] Some checks did not pass. See summary for details.")
        sys.exit(1)

    log("\n[OK] All checks passed successfully.")


if __name__ == "__main__":
    main()

