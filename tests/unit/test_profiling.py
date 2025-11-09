from __future__ import annotations

from tiny_backprop.runtime.profiling import Profiler


def test_profiler_records_stats() -> None:
    profiler = Profiler()

    profiler.record_memory(100)
    profiler.record_memory(50)  # should not reduce peak
    profiler.record_flops(1_000)
    profiler.record_flops(500)
    profiler.record_event("forward", 1.5)
    profiler.record_event("forward", 0.5)

    stats = profiler.snapshot()
    assert stats.peak_memory_bytes == 100
    assert stats.recompute_flops == 1500
    assert stats.recompute_steps == 2
    assert stats.events["forward"] == 2.0

