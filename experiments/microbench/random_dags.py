"""
Generate synthetic DAGs and evaluate frontier width heuristics.
"""

from __future__ import annotations

import argparse
import random
from typing import Tuple

from tiny_backprop.analysis import evaluate_hcp, summarize_frontier, width_heuristics
from tiny_backprop.graph.ir import Graph, Node


PROFILES = {
    "small": {
        "nodes": 64,
        "max_parents": 3,
        "size_min": 64_000,
        "size_max": 512_000,
    },
    "medium": {
        "nodes": 192,
        "max_parents": 4,
        "size_min": 128_000,
        "size_max": 1_024_000,
    },
    "large": {
        "nodes": 384,
        "max_parents": 6,
        "size_min": 256_000,
        "size_max": 2_048_000,
    },
}


def build_random_dag(
    num_nodes: int,
    max_parents: int,
    size_range: Tuple[int, int],
    *,
    seed: int,
) -> Graph:
    rng = random.Random(seed)
    graph = Graph()

    for idx in range(num_nodes):
        name = f"n{idx}"
        if idx == 0:
            inputs = []
        else:
            parent_candidates = list(range(idx))
            rng.shuffle(parent_candidates)
            num_parents = rng.randint(1, min(max_parents, idx))
            inputs = [f"n{p}" for p in parent_candidates[:num_parents]]
        size = rng.randint(*size_range)
        node = Node(name=name, op="linear", inputs=inputs, outputs_size=size)
        graph.add_node(node)

    graph.inputs = ["n0"]
    graph.outputs = [f"n{num_nodes - 1}"]
    graph.validate()
    return graph


def analyze_graph(graph: Graph) -> None:
    order_stats = width_heuristics.evaluate_orders(graph)
    best = order_stats[0]
    summarize_frontier(graph, best.order)
    hcp_eval = evaluate_hcp(graph)
    print("=== Graph Diagnostics ===")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Best heuristic: {best.strategy} (peak={best.max_live/1e6:.2f} MB)")
    print(f"HCP-like: {hcp_eval.is_hcp_like} (frontier={hcp_eval.frontier_width/1e6:.2f} MB)")
    print("Notes:")
    for note in hcp_eval.notes:
        print(f"  - {note}")


def _apply_profile(args: argparse.Namespace) -> argparse.Namespace:
    profile_cfg = PROFILES.get(args.profile)
    if not profile_cfg:
        return args
    for key, value in profile_cfg.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES.keys(), default="medium")
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--max-parents", type=int, default=None)
    parser.add_argument("--size-min", type=int, default=None)
    parser.add_argument("--size-max", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args = _apply_profile(args)
    if args.nodes is None:
        args.nodes = 64
    if args.max_parents is None:
        args.max_parents = 3
    if args.size_min is None:
        args.size_min = 64_000
    if args.size_max is None:
        args.size_max = 512_000
    return args


def main() -> None:
    args = parse_args()
    graph = build_random_dag(
        num_nodes=args.nodes,
        max_parents=args.max_parents,
        size_range=(args.size_min, args.size_max),
        seed=args.seed,
    )
    analyze_graph(graph)


if __name__ == "__main__":
    main()

