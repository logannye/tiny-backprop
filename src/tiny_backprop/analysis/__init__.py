"""
Graph analysis utilities for height-compressed backprop.

Key responsibilities:
- Estimate frontier width (activation 'live set' over time).
- Diagnose whether a graph is "height-compressible" (HCP-like).
- Provide lower bounds to evaluate schedules against.
"""

from .frontier_width import FrontierSnapshot, FrontierSummary, compute_frontier_width, frontier_profile, summarize_frontier
from .hcp_criteria import HCPEvaluation, evaluate_hcp
from . import lower_bounds
from . import width_heuristics
from .report import GraphAnalysisReport, analyze_graph

__all__ = [
    "FrontierSnapshot",
    "FrontierSummary",
    "compute_frontier_width",
    "frontier_profile",
    "summarize_frontier",
    "HCPEvaluation",
    "evaluate_hcp",
    "lower_bounds",
    "width_heuristics",
    "GraphAnalysisReport",
    "analyze_graph",
]
