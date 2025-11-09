"""
Graph intermediate representation (IR) and utilities.

This package defines a simple, framework-agnostic DAG representation
used by the planner:

- `Graph` and `Node` (see `ir.py`)
- Builders from framework graphs (Torch FX, JAXPR, etc.)
- Topological ordering and visualization helpers.
"""

from .ir import Graph, Node
from . import builders
from . import topo
from . import visualize

__all__ = [
    "Graph",
    "Node",
    "builders",
    "topo",
    "visualize",
]
