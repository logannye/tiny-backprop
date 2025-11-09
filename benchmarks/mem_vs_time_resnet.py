"""
Benchmark memory/time trade-offs for a ResNet-like model across execution modes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
        "batch_size": 4,
        "image_size": 128,
        "block_size": 8,
    },
    "medium": {
        "batch_size": 8,
        "image_size": 224,
        "block_size": 16,
    },
    "large": {
        "batch_size": 12,
        "image_size": 320,
        "block_size": 24,
    },
}


def tiny_resnet18(num_classes: int = 10) -> nn.Module:
    # Use torchvision if available; otherwise a trivial CNN stub.
    try:
        from torchvision.models import resnet18

        model = resnet18(num_classes=num_classes)
        return model
    except Exception:
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _instantiate_model(
    mode: str,
    *,
    seed: int,
    device: torch.device,
    block_size: int,
) -> nn.Module:
    torch.manual_seed(seed)
    base = tiny_resnet18()
    base_state = base.state_dict()

    if mode == "naive":
        model = tiny_resnet18()
        model.load_state_dict(base_state)
        return model.to(device)

    if mode == "tiny":
        inner = tiny_resnet18()
        inner.load_state_dict(base_state)
        config = HCConfig(block_size=block_size)
        model = HCModule(inner, config=config)
        return model.to(device)

    raise ValueError(f"Unknown mode: {mode}")


def _prepare_batch(
    device: torch.device, batch_size: int, image_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(batch_size, 3, image_size, image_size, device=device).requires_grad_()
    y = torch.randint(0, 10, (batch_size,), device=device)
    return x, y


def run_benchmark(args: argparse.Namespace) -> list[BenchmarkResult]:
    device = _select_device()
    x, targets = _prepare_batch(device, args.batch_size, args.image_size)
    criterion = nn.CrossEntropyLoss()

    results: list[BenchmarkResult] = []
    for mode in args.modes:
        for trial in range(args.trials):
            seed = args.seed + trial
            model = _instantiate_model(
                mode,
                seed=seed,
                device=device,
                block_size=args.block_size,
            )

            def forward_fn(model=model) -> torch.Tensor:
                model.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, targets)
                return loss

            def backward_fn(loss: torch.Tensor, model=model) -> None:
                if isinstance(model, HCModule):
                    model.backward(loss)
                else:
                    loss.backward()

            result = run_single_trial(
                "resnet",
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
    parser.add_argument("--modes", nargs="+", default=["naive", "tiny"])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--profile", choices=PROFILES.keys(), default="medium")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--export", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()
    args = _apply_profile(args)
    if args.batch_size is None:
        args.batch_size = 8
    if args.image_size is None:
        args.image_size = 224
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
