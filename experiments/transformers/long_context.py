"""
Sweep context lengths to understand memory scaling with tiny-backprop.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import torch
from torch import nn

from benchmarks.utils import (
    BenchmarkResult,
    export_json,
    format_summary_table,
    run_single_trial,
    summarize,
)
from tiny_backprop.integration.torch import HCConfig, HCModule


PROFILES = {
    "small": {
        "seq_lens": [128, 256, 384],
        "d_model": 192,
        "depth": 4,
        "block_size": 16,
        "batch_size": 2,
        "trials": 1,
    },
    "medium": {
        "seq_lens": [256, 512, 1024, 1536],
        "d_model": 256,
        "depth": 6,
        "block_size": 32,
        "batch_size": 2,
        "trials": 2,
    },
    "large": {
        "seq_lens": [512, 1024, 2048, 4096],
        "d_model": 384,
        "depth": 8,
        "block_size": 48,
        "batch_size": 2,
        "trials": 2,
    },
}


class ContextTransformer(nn.Module):
    def __init__(self, d_model: int = 256, depth: int = 6, nhead: int = 8):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=4 * d_model,
                    batch_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _instantiate_models(
    d_model: int,
    depth: int,
    block_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, nn.Module]:
    torch.manual_seed(seed)
    base = ContextTransformer(d_model=d_model, depth=depth)
    base_state = base.state_dict()

    naive = ContextTransformer(d_model=d_model, depth=depth).to(device)
    naive.load_state_dict(base_state)

    inner = ContextTransformer(d_model=d_model, depth=depth)
    inner.load_state_dict(base_state)
    tiny = HCModule(inner, config=HCConfig(block_size=block_size)).to(device)

    return {"naive": naive, "tiny": tiny}


def sweep(
    seq_lengths: Iterable[int],
    *,
    d_model: int,
    depth: int,
    block_size: int,
    batch_size: int,
    trials: int,
    seed: int,
) -> List[BenchmarkResult]:
    device = _select_device()
    results: list[BenchmarkResult] = []

    for seq_len in seq_lengths:
        torch.manual_seed(0)
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        y_true = torch.randn_like(x)

        for trial in range(trials):
            models = _instantiate_models(
                d_model=d_model,
                depth=depth,
                block_size=block_size,
                device=device,
                seed=seed + trial,
            )

            for mode, model in models.items():
                def forward_fn() -> torch.Tensor:
                    model.zero_grad(set_to_none=True)
                    out = model(x)
                    return torch.nn.functional.mse_loss(out, y_true)

                def backward_fn(loss: torch.Tensor) -> None:
                    if isinstance(model, HCModule):
                        model.backward(loss)
                    else:
                        loss.backward()

                result = run_single_trial(
                    benchmark_name=f"transformer_seq_{seq_len}",
                    mode=mode,
                    trial=trial,
                    device=device,
                    forward_fn=forward_fn,
                    backward_fn=backward_fn,
                )
                results.append(result)

            # free models to avoid GPU accumulation
            for mdl in models.values():
                del mdl

    return results


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
    parser.add_argument("--seq-lens", nargs="+", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--export", type=Path, help="Optional JSON export path.")
    args = parser.parse_args()
    args = _apply_profile(args)
    if args.seq_lens is None:
        args.seq_lens = [128, 256, 512, 1024]
    if args.d_model is None:
        args.d_model = 256
    if args.depth is None:
        args.depth = 6
    if args.block_size is None:
        args.block_size = 32
    if args.batch_size is None:
        args.batch_size = 2
    if args.trials is None:
        args.trials = 2
    return args


def main() -> None:
    args = parse_args()
    results = sweep(
        seq_lengths=args.seq_lens,
        d_model=args.d_model,
        depth=args.depth,
        block_size=args.block_size,
        batch_size=args.batch_size,
        trials=args.trials,
        seed=args.seed,
    )
    summary = summarize(results)
    print(format_summary_table(summary))

    if args.export:
        export_json(results, args.export)
        print(f"\nSaved raw results to {args.export}")


if __name__ == "__main__":
    main()
