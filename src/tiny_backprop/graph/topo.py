"""
Topological order selection & simple width-minimizing heuristics.
"""

from __future__ import annotations

from typing import List

from .ir import Graph, Node


def default_topological_order(graph: Graph) -> List[Node]:
    """
    Use the built-in topo sort for now.
    Later: plug in heuristics to reduce frontier width.
    """
    return graph.topological_sort()
