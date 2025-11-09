from __future__ import annotations

from benchmarks.utils import BenchmarkResult, format_summary_table, summarize


def test_summarize_and_format_table() -> None:
    results = [
        BenchmarkResult(
            "demo",
            "naive",
            0,
            wall_time_s=0.5,
            peak_mem_bytes=1000,
            peak_mem_reserved_bytes=1100,
            gpu_utilization_pct=None,
            loss_value=1.0,
        ),
        BenchmarkResult(
            "demo",
            "naive",
            1,
            wall_time_s=0.7,
            peak_mem_bytes=1200,
            peak_mem_reserved_bytes=1300,
            gpu_utilization_pct=50.0,
            loss_value=1.5,
        ),
        BenchmarkResult(
            "demo",
            "tiny",
            0,
            wall_time_s=0.8,
            peak_mem_bytes=800,
            peak_mem_reserved_bytes=900,
            gpu_utilization_pct=55.0,
            loss_value=1.2,
        ),
    ]

    summary = summarize(results)
    table = format_summary_table(summary)

    assert any(row["mode"] == "naive" for row in summary)
    assert "mode" in table

