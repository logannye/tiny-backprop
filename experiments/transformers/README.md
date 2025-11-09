# Transformer Experiments

This directory contains experiments that apply **tiny-backprop** to
Transformer-style models (GPT-style decoders, encoders, long-context variants).

The goal: **show, from first principles, how height-compressed backprop reduces
activation memory for realistic, DAG-shaped architectures without breaking
performance or gradients.**

---

## What we're testing

Transformers are an ideal testbed because:

1. **They are deep, regular, and expensive in activations.**
   - Depth `L`: stacked attention + MLP blocks.
   - Width `w`: hidden size × sequence length × number of heads.
   - Naive training stores activations for each block → `O(L·w)` memory.

2. **Their computation graph is almost a chain, but with structure.**
   - Residual connections.
   - LayerNorm, attention, MLP branches.
   - All of this forms a **DAG with bounded frontier width** for typical designs.
   - This is exactly the “height-compressible” regime tiny-backprop targets.

3. **They matter commercially.**
   - Any significant reduction in memory or recompute overhead is immediately useful.

---

## Files

### `gpt2_mem_bench.py`

- Baseline: standard PyTorch-style GPT-2 or similar decoder-only stack.
- Compares:
  - naive autograd,
  - built-in gradient checkpointing (if enabled),
  - tiny-backprop planned schedule (once implemented).
- Metrics:
  - peak activation memory,
  - wall-clock time per step,
  - recompute overhead.

### `long_context.py`

- Stress-test on **sequence length** and **depth**.
- Explores:
  - how memory scales as `L` (layers) and `T` (tokens) grow,
  - when tiny-backprop turns infeasible contexts into feasible ones.
- Intended to produce plots like:
  - memory vs sequence length,
  - memory vs layers for fixed hardware.

---

## How this directory fits into the repo

These experiments demonstrate:

- How we:
  1. Trace the model into a computation graph.
  2. Analyze frontier width and structure.
  3. Build a checkpoint & replay schedule.
  4. Run backprop with reduced memory.

- That we can:
  - Preserve exact gradients,
  - Keep recomputation bounded,
  - Achieve predictable memory savings (e.g. 2–10× vs naive),
  - Beat or match hand-tuned checkpoint heuristics.

In other words: **this is the flagship “does it work on real models?” evidence
for tiny-backprop.**
