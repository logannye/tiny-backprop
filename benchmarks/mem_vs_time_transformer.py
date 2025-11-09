"""
Benchmark memory/time trade-offs for a modest Transformer across execution modes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

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
        "batch_size": 4,
        "seq_len": 64,
        "d_model": 128,
        "nhead": 4,
        "layers": 2,
        "block_size": 8,
    },
    "medium": {
        "batch_size": 8,
        "seq_len": 256,
        "d_model": 192,
        "nhead": 6,
        "layers": 6,
        "block_size": 16,
    },
    "large": {
        "batch_size": 12,
        "seq_len": 512,
        "d_model": 256,
        "nhead": 8,
        "layers": 8,
        "block_size": 32,
    },
}


class TinyTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        *,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=4 * d_model,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and len(self.layers) > 0:
            segments = len(self.layers)
            x = checkpoint_sequential(list(self.layers), segments=segments, input=x)
        else:
            for layer in self.layers:
                x = layer(x)
        x = self.norm(x)
        return self.proj(x)


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _instantiate_model(
    mode: str,
    *,
    seed: int,
    device: torch.device,
    d_model: int,
    nhead: int,
    layers: int,
    block_size: int,
) -> nn.Module:
    torch.manual_seed(seed)
    base = TinyTransformer(d_model=d_model, nhead=nhead, num_layers=layers)
    base_state = base.state_dict()

    if mode == "naive":
        model = TinyTransformer(
            d_model=d_model, nhead=nhead, num_layers=layers, use_checkpoint=False
        )
        model.load_state_dict(base_state)
        return model.to(device)

    if mode == "checkpoint":
        model = TinyTransformer(
            d_model=d_model, nhead=nhead, num_layers=layers, use_checkpoint=True
        )
        model.load_state_dict(base_state)
        return model.to(device)

    if mode == "tiny":
        inner = TinyTransformer(
            d_model=d_model, nhead=nhead, num_layers=layers, use_checkpoint=False
        )
        inner.load_state_dict(base_state)
        config = HCConfig(block_size=block_size)
        model = HCModule(inner, config=config)
        return model.to(device)

    raise ValueError(f"Unknown mode: {mode}")


def _prepare_batch(
    device: torch.device, batch_size: int, seq_len: int, d_model: int
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(batch_size, seq_len, d_model, device=device).requires_grad_()
    y_true = torch.randn_like(x)
    return x, y_true


def run_benchmark(args: argparse.Namespace) -> list[BenchmarkResult]:
    device = _select_device()
    x, y_true = _prepare_batch(
        device, batch_size=args.batch_size, seq_len=args.seq_len, d_model=args.d_model
    )

    results: list[BenchmarkResult] = []

    for mode in args.modes:
        for trial in range(args.trials):
            seed = args.seed + trial
            model = _instantiate_model(
                mode,
                seed=seed,
                device=device,
                d_model=args.d_model,
                nhead=args.nhead,
                layers=args.layers,
                block_size=args.block_size,
            )

            def forward_fn(model=model) -> torch.Tensor:
                model.zero_grad(set_to_none=True)
                out = model(x)
                loss = torch.nn.functional.mse_loss(out, y_true)
                return loss

            def backward_fn(loss: torch.Tensor, model=model) -> None:
                if isinstance(model, HCModule):
                    model.backward(loss)
                else:
                    loss.backward()

            result = run_single_trial(
                "transformer",
                mode,
                trial,
                device=device,
                forward_fn=forward_fn,
                backward_fn=backward_fn,
            )
            results.append(result)
            del model

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
    parser.add_argument("--modes", nargs="+", default=["naive", "checkpoint", "tiny"])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--profile", choices=PROFILES.keys(), default="medium")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--export", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()
    args = _apply_profile(args)
    # Fallback to legacy defaults if profile omitted values
    if args.batch_size is None:
        args.batch_size = 8
    if args.seq_len is None:
        args.seq_len = 128
    if args.d_model is None:
        args.d_model = 128
    if args.nhead is None:
        args.nhead = 4
    if args.layers is None:
        args.layers = 4
    if args.block_size is None:
        args.block_size = 16
    return args


def main() -> None:
    args = parse_args()
    results = run_benchmark(args)
    summary = summarize(results)
    print(format_summary_table(summary))

    if args.export:
        export_json(results, args.export)
        print(f"\nSaved raw results to {args.export}")


if __name__ == "__main__":
    main()
