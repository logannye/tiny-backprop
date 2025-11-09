# Theory Primer

This page sketches the mathematical grounding for **height-compressed
backpropagation (HCB)**. The goal is to connect the implemented planner to
classical results on time–space trade-offs for reversible computations, and to
highlight assumptions necessary for the guarantees to hold.

## Reverse-Mode AD as a Pebbling Problem

Consider a forward computation represented by a directed acyclic graph (DAG)
`G = (V, E)` with topological order `v1, …, vN`. Reverse-mode automatic
differentiation (a.k.a. backprop) needs the value of each node when back-propagating
gradients through its outgoing edges.

The naive strategy retains all activations, leading to peak memory
`O(sum_{v in V} size(v))`. Classical **pebbling games** rephrase the problem:

- A pebble on `v` represents having the forward activation of `v`.
- You can place a pebble on `v` once all predecessors have pebbles.
- You can remove pebbles to free memory, but must recompute `v` later if needed.

The objective is to minimise the maximum number of pebbles (memory) while keeping
total recomputation bounded.

## Height-Compression Intuition

If the DAG has bounded “live width” (at any forward step, only `w` nodes remain
live), it is possible to **compress** the effective height by partitioning the order
into blocks of length `B` and retaining only boundary activations. Interior nodes are
recomputed on demand during backward.

For chain-like computations (e.g., Revolve schedule) the optimal trade-off is
`memory ~ O(sqrt(N))` with `O(sqrt(N))` recompute factor. HCB generalises the idea to
height-compressible DAGs: apply the same blocking and hierarchical replay to graphs
with limited fan-out / skip lengths.

### Informal Guarantee

Let:

- `L` be the depth (number of nodes in the chosen topological order).
- `w` be the maximum live activation size in that order (frontier width).
- `B` be the block size chosen by the planner.

We obtain:

- **Peak activation memory** ≈ `O(w * sqrt(L))` by selecting `B ~ sqrt(L)` and
  retaining only `O(sqrt(L))` block boundaries.
- **Recomputation overhead** bounded by `O(sqrt(L))` because each block interior is
  replayed at most logarithmically many times as we traverse the interval tree.

The planner in this repository implements the interval-tree schedule that realises
this trade-off up to constant factors.

## Core Algorithm Outline

1. **Topological order**: choose an order with small frontier width (heuristics or
   user-supplied).
2. **Block partitioning**: divide nodes into consecutive blocks of size `B`.
3. **Balanced tree**: build a binary tree whose leaves correspond to blocks; internal
   nodes cover contiguous intervals.
4. **Checkpoint selection**: store activations at block boundaries only.
5. **Replay**: during backward, traverse the tree in post-order:
   - Recompute the forward pass for an interval if necessary (ensured by checkpoints).
   - Execute backward for that interval and free interior activations.

This yields a deterministic schedule that can be validated statically (no runtime
heuristics).

## Structural Assumptions

Height compression performs best when:

- The DAG has **limited fan-out** (residual connections are short).
- There is a **dominant depth dimension** (layers) versus exponential branching.
- Activations are **recomputable** at acceptable cost (no massive stochastic
  branches).

Pathological cases (e.g., all-to-all attention over the entire past) may drive the
frontier width close to the total activation size, at which point tiny-backprop falls
back to near-naive memory.

## Relationship to Lower Bounds

- `frontier_based_lower_bound` reports the minimum achievable peak if you could pick
  the best possible topological order (subject to heuristic search).
- `naive_lower_bound` (max activation) is trivial but useful sanity check: any plan
  using less than the largest single activation is impossible.
- The **gap** between the planner’s peak memory and the lower bound tracks how close
  we are to optimal. Experiments can plot this gap across architectures.

## References

- Griewank & Walther (2000). *Algorithm 799: Revolve: an implementation of checkpointing for the reverse or adjoint mode of computational differentiation.* ACM TOMS.
- Bennett (1989). *Time-space trade-offs for reversible computation.* SIAM J. Comput.
- Sefair et al. (2024). *Optimal Checkpoint Schedules for DAG-Shaped Neural Networks.* (Hypothetical; slot in relevant contemporary work.)

While the current codebase does not yet include formal proofs, the scheduling
strategy mirrors those classical constructions and is designed to be amenable to
future formal verification.

