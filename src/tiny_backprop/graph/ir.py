from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


@dataclass
class Node:
    """Framework-agnostic node in the computation DAG."""

    name: str
    op: str
    inputs: List[str] = field(default_factory=list)
    outputs_size: int = 0
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outputs_size < 0:
            raise ValueError(f"Node `{self.name}` has negative activation size.")

    def copy(self) -> "Node":
        return Node(
            name=self.name,
            op=self.op,
            inputs=list(self.inputs),
            outputs_size=self.outputs_size,
            attrs=dict(self.attrs),
        )


@dataclass
class Graph:
    """
    Framework-agnostic representation of a forward computation graph.
    """
    nodes: Dict[str, Node] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: Node, *, allow_overwrite: bool = False) -> None:
        if not allow_overwrite and node.name in self.nodes:
            raise ValueError(f"Duplicate node name: {node.name}")
        self.nodes[node.name] = node

    def get_node(self, name: str) -> Node:
        return self.nodes[name]

    def predecessors(self, name: str) -> List[Node]:
        node = self.get_node(name)
        return [self.get_node(i) for i in node.inputs]

    def successors(self, name: str) -> List[Node]:
        return [n for n in self.nodes.values() if name in n.inputs]

    def validate(self) -> None:
        """
        Validate structural soundness:
        - inputs/outputs reference existing nodes
        - graph is acyclic
        - node dependencies exist
        """
        missing_edges: List[str] = []
        for node in self.nodes.values():
            for inp in node.inputs:
                if inp not in self.nodes:
                    missing_edges.append(f"{node.name} -> {inp}")
        if missing_edges:
            raise ValueError(
                "Graph references unknown predecessors:\n" + "\n".join(missing_edges)
            )

        for name in self.inputs:
            if name not in self.nodes:
                raise ValueError(f"Declared input `{name}` not found in graph nodes.")
        for name in self.outputs:
            if name not in self.nodes:
                raise ValueError(f"Declared output `{name}` not found in graph nodes.")

        # Will raise if a cycle exists or dependencies missing.
        self.topological_sort()

    def topological_sort(self) -> List[Node]:
        """
        Standard Kahn topo-sort.

        Returns:
            List of nodes in a valid topological order.
        """
        indeg: Dict[str, int] = {name: 0 for name in self.nodes}
        succ: Dict[str, List[str]] = {name: [] for name in self.nodes}

        for node in self.nodes.values():
            for parent in node.inputs:
                if parent not in self.nodes:
                    raise KeyError(
                        f"Node `{node.name}` depends on unknown predecessor `{parent}`."
                    )
                indeg[node.name] += 1
                succ[parent].append(node.name)

        ready = deque(sorted([name for name, deg in indeg.items() if deg == 0]))
        order: List[str] = []

        while ready:
            current = ready.popleft()
            order.append(current)
            for child in sorted(succ[current]):
                indeg[child] -= 1
                if indeg[child] == 0:
                    ready.append(child)

        if len(order) != len(self.nodes):
            raise ValueError("Graph has cycles or is malformed.")

        return [self.nodes[n] for n in order]

    def live_frontier_sizes(
        self, order: Optional[Sequence[Node]] = None
    ) -> List[int]:
        """
        Compute live frontier size over a given topo order (naive version).

        Returns:
            List of live sizes per step.
        """
        if order is None:
            order = self.topological_sort()
        else:
            # Convert to concrete Node list in case caller passes names.
            order = [self.nodes[n.name] if isinstance(n, Node) else self.nodes[n] for n in order]  # type: ignore[index]

        live: Set[str] = set()
        sizes: List[int] = []
        needed_by: Dict[str, int] = {n.name: 0 for n in order}
        for n in order:
            for i in n.inputs:
                needed_by[i] += 1

        for n in order:
            # Node's output becomes live
            live.add(n.name)
            sizes.append(sum(self.nodes[x].outputs_size for x in live))

            # Decrement usage of its inputs
            for i in n.inputs:
                needed_by[i] -= 1
                if needed_by[i] == 0 and i in live:
                    live.remove(i)

        return sizes

    def total_activation_size(self) -> int:
        """Sum of all node activation sizes."""
        return sum(node.outputs_size for node in self.nodes.values())

    def induced_subgraph(self, names: Iterable[str]) -> "Graph":
        """Create a shallow copy containing only the selected node names."""
        sub = Graph(metadata=dict(self.metadata))
        for name in names:
            sub.add_node(self.get_node(name).copy())
        sub.inputs = [n for n in self.inputs if n in sub.nodes]
        sub.outputs = [n for n in self.outputs if n in sub.nodes]
        return sub
