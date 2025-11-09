"""
Benchmark a lightweight U-Net under naive vs height-compressed execution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

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
        "base_channels": 32,
        "depth": 2,
        "block_size": 8,
        "batch_size": 1,
        "image_size": 128,
        "trials": 1,
    },
    "medium": {
        "base_channels": 64,
        "depth": 3,
        "block_size": 16,
        "batch_size": 2,
        "image_size": 192,
        "trials": 2,
    },
    "large": {
        "base_channels": 96,
        "depth": 4,
        "block_size": 24,
        "batch_size": 2,
        "image_size": 256,
        "trials": 3,
    },
}


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    def __init__(self, base_channels: int = 64, depth: int = 3):
        super().__init__()
        channels = [base_channels * (2**i) for i in range(depth)]

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_ch = 3
        for ch in channels:
            self.enc_blocks.append(ConvBlock(in_ch, ch))
            self.downs.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            in_ch = ch

        self.bottleneck = ConvBlock(channels[-1], channels[-1])

        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for ch in reversed(channels):
            self.ups.append(nn.ConvTranspose2d(in_ch, ch, 2, stride=2))
            self.dec_blocks.append(ConvBlock(ch * 2, ch))
            in_ch = ch

        self.head = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        for block, down in zip(self.enc_blocks, self.downs):
            x = block(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up, block in zip(self.ups, self.dec_blocks):
            x = up(x)
            skip = skips.pop()
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return self.head(x)


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _instantiate_models(
    *,
    base_channels: int,
    depth: int,
    block_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, nn.Module]:
    torch.manual_seed(seed)
    base = TinyUNet(base_channels=base_channels, depth=depth)
    base_state = base.state_dict()

    naive = TinyUNet(base_channels=base_channels, depth=depth).to(device)
    naive.load_state_dict(base_state)

    inner = TinyUNet(base_channels=base_channels, depth=depth)
    inner.load_state_dict(base_state)
    tiny = HCModule(inner, config=HCConfig(block_size=block_size)).to(device)

    return {"naive": naive, "tiny": tiny}


def run_benchmark(args: argparse.Namespace) -> list[BenchmarkResult]:
    device = _select_device()
    torch.manual_seed(0)
    inputs = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)

    results: list[BenchmarkResult] = []
    for trial in range(args.trials):
        models = _instantiate_models(
            base_channels=args.base_channels,
            depth=args.depth,
            block_size=args.block_size,
            device=device,
            seed=args.seed + trial,
        )

        for mode, model in models.items():
            def forward_fn() -> torch.Tensor:
                model.zero_grad(set_to_none=True)
                output = model(inputs)
                return torch.nn.functional.mse_loss(output, inputs)

            def backward_fn(loss: torch.Tensor) -> None:
                if isinstance(model, HCModule):
                    model.backward(loss)
                else:
                    loss.backward()

            result = run_single_trial(
                benchmark_name="unet",
                mode=mode,
                trial=trial,
                device=device,
                forward_fn=forward_fn,
                backward_fn=backward_fn,
            )
            results.append(result)

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
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--export", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()
    args = _apply_profile(args)
    if args.base_channels is None:
        args.base_channels = 64
    if args.depth is None:
        args.depth = 3
    if args.block_size is None:
        args.block_size = 32
    if args.batch_size is None:
        args.batch_size = 2
    if args.image_size is None:
        args.image_size = 256
    if args.trials is None:
        args.trials = 3
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

