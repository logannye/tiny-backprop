from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import List, Sequence

from tiny_backprop.graph.ir import Graph, Node
from tiny_backprop.graph.topo import default_topological_order


@dataclass(frozen=True)
class FrontierSnapshot:
    step: int
    node: str
    live_memory: int
    entering: tuple[str, ...]
    exiting: tuple[str, ...]


@dataclass(frozen=True)
class FrontierSummary:
    snapshots: List[FrontierSnapshot]
    max_live: int
    argmax_step: int
    average_live: float
    final_live: int


def _normalize_order(graph: Graph, order: Sequence[Node] | Sequence[str] | None) -> List[Node]:
    if order is None:
        return default_topological_order(graph)
    if not order:
        return []
    first = order[0]
    if isinstance(first, Node):
        return list(order)  # type: ignore[return-value]
    return [graph.get_node(name) for name in order]  # type: ignore[arg-type]


def frontier_profile(graph: Graph, order: Sequence[Node] | Sequence[str] | None = None) -> List[FrontierSnapshot]:
    resolved_order = _normalize_order(graph, order)

    needed_by: dict[str, int] = {node.name: 0 for node in resolved_order}
    for node in resolved_order:
        for inp in node.inputs:
            needed_by[inp] = needed_by.get(inp, 0) + 1

    live: set[str] = set()
    live_memory = 0
    snapshots: List[FrontierSnapshot] = []

    for step, node in enumerate(resolved_order):
        entering: List[str] = []
        exiting: List[str] = []

        if node.name not in live:
            live.add(node.name)
            live_memory += node.outputs_size
            entering.append(node.name)

        for inp in node.inputs:
            needed_by[inp] -= 1
            if needed_by[inp] <= 0 and inp in live:
                live.remove(inp)
                live_memory -= graph.get_node(inp).outputs_size
                exiting.append(inp)

        snapshots.append(
            FrontierSnapshot(
                step=step,
                node=node.name,
                live_memory=live_memory,
                entering=tuple(entering),
                exiting=tuple(exiting),
            )
        )

    return snapshots


def summarize_frontier(graph: Graph, order: Sequence[Node] | Sequence[str] | None = None) -> FrontierSummary:
    snapshots = frontier_profile(graph, order)

    if not snapshots:
        return FrontierSummary(
            snapshots=[],
            max_live=0,
            argmax_step=-1,
            average_live=0.0,
            final_live=0,
        )

    max_live = max(s.live_memory for s in snapshots)
    argmax_step = next(i for i, s in enumerate(snapshots) if s.live_memory == max_live)
    avg_live = mean(s.live_memory for s in snapshots)
    final_live = snapshots[-1].live_memory

    return FrontierSummary(
        snapshots=snapshots,
        max_live=max_live,
        argmax_step=argmax_step,
        average_live=avg_live,
        final_live=final_live,
    )


def compute_frontier_width(graph: Graph, order: Sequence[Node] | Sequence[str] | None = None) -> int:
    """
    Compute maximum live frontier size (sum of activation sizes) for given order.
    """
    summary = summarize_frontier(graph, order)
    return summary.max_live
