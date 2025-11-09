"""
Explore memory scaling for progressively deeper residual networks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

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


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return F.relu(out, inplace=True)


class DeepResNet(nn.Module):
    def __init__(self, channels: int = 64, depth: int = 18, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(depth)]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _instantiate_models(
    depth: int,
    *,
    channels: int,
    block_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, nn.Module]:
    torch.manual_seed(seed)
    base = DeepResNet(channels=channels, depth=depth)
    base_state = base.state_dict()

    naive = DeepResNet(channels=channels, depth=depth).to(device)
    naive.load_state_dict(base_state)

    inner = DeepResNet(channels=channels, depth=depth)
    inner.load_state_dict(base_state)
    tiny = HCModule(inner, config=HCConfig(block_size=block_size)).to(device)

    return {"naive": naive, "tiny": tiny}


def sweep_depths(
    depths: Iterable[int],
    *,
    channels: int,
    block_size: int,
    batch_size: int,
    image_size: int,
    trials: int,
    seed: int,
) -> List[BenchmarkResult]:
    device = _select_device()
    criterion = nn.CrossEntropyLoss()
    results: list[BenchmarkResult] = []

    for depth in depths:
        torch.manual_seed(0)
        inputs = torch.randn(batch_size, 3, image_size, image_size, device=device)
        targets = torch.randint(0, 10, (batch_size,), device=device)

        for trial in range(trials):
            models = _instantiate_models(
                depth=depth,
                channels=channels,
                block_size=block_size,
                device=device,
                seed=seed + trial,
            )

            for mode, model in models.items():
                def forward_fn() -> torch.Tensor:
                    model.zero_grad(set_to_none=True)
                    logits = model(inputs)
                    return criterion(logits, targets)

                def backward_fn(loss: torch.Tensor) -> None:
                    if isinstance(model, HCModule):
                        model.backward(loss)
                    else:
                        loss.backward()

                result = run_single_trial(
                    benchmark_name=f"resnet_depth_{depth}",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", nargs="+", type=int, default=[8, 18, 26])
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--export", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = sweep_depths(
        depths=args.depths,
        channels=args.channels,
        block_size=args.block_size,
        batch_size=args.batch_size,
        image_size=args.image_size,
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

