# Microbenchmarks & Synthetic DAG Experiments

This directory contains **controlled experiments** to understand
tiny-backprop’s behavior from first principles:

- On idealized computation graphs.
- On random or adversarial DAGs.
- Against theoretical lower bounds.

Think of this as the **“lab environment”** for the algorithm.

---

## Why microbenchmarks matter

Before trusting a scheduler on massive real models, we want to:

1. Validate correctness mechanically.
2. Compare:
   - planned memory usage vs.
   - simple lower bounds (e.g. frontier width, pebbling arguments).
3. Explore failure modes:
   - graphs that are *not* height-compressible,
   - pathological long-range dependencies,
   - extremely wide layers.

This directory gives you those knobs without the noise of framework internals.

---

## Files

### `random_dags.py`

- Generates random DAGs under various regimes:
  - skinny chains,
  - layered graphs,
  - high cross-link densities (non-HCP-like),
  - configurable node sizes.
- Runs the tiny-backprop planner on each.
- Reports:
  - estimated frontier width,
  - planned peak memory,
  - recomputation overhead.
- Purpose:
  - see when the algorithm shines,
  - see when it degrades (correctly) to “no real savings”.

### `compare_checkpointing.py`

- Compares multiple strategies on the same synthetic graphs:
  - naive (store everything),
  - simple periodic checkpointing,
  - Revolve-style (for chain-like),
  - tiny-backprop’s interval-tree schedule.
- Outputs:
  - memory vs time trade-off curves.
- Purpose:
  - empirically show tiny-backprop is:
    - competitive with optimal known schedules on chains,
    - strictly better than naive heuristics on structured DAGs.

### `lower_bound_gap.py`

- Uses analytic or heuristic lower bounds:
  - min possible memory from frontier width / cutwidth approximations.
- Measures:
  - `gap = (memory_used_by_tiny_backprop / lower_bound)`.
- Purpose:
  - quantify “near-optimality”:
    - is the gap close to 1× on nice graphs?
    - how bad can it get on adversarial graphs?

---

## How to use this directory

For ML engineers and researchers:

- Start here if you want to **understand the algorithm itself**:
  - how the scheduler behaves as graph structure changes,
  - whether it matches your intuition from complexity / pebbling theory.
- Use these scripts to:
  - debug new scheduling ideas,
  - validate theoretical claims,
  - generate figures for papers and docs.

Once you’re confident from microbenchmarks,
move on to `transformers/`, `vision/`, and `diffusion/`
to see how it behaves on real-world architectures.
