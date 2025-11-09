"""
Compare GPT-style decoder memory/time across naive, checkpoint, and tiny-backprop.
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
        "batch_size": 2,
        "seq_len": 64,
        "vocab_size": 2000,
        "d_model": 192,
        "layers": 4,
        "block_size": 16,
    },
    "medium": {
        "batch_size": 4,
        "seq_len": 256,
        "vocab_size": 4000,
        "d_model": 256,
        "layers": 8,
        "block_size": 32,
    },
    "large": {
        "batch_size": 6,
        "seq_len": 512,
        "vocab_size": 8000,
        "d_model": 384,
        "layers": 12,
        "block_size": 48,
    },
}


class TinyGPT2(nn.Module):
    # Placeholder; swap in HuggingFace GPT-2 later.
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_layer: int = 4,
        *,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=4,
                    dim_feedforward=4 * d_model,
                    batch_first=True,
                )
                for _ in range(n_layer)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        if self.use_checkpoint and len(self.layers) > 0:
            h = checkpoint_sequential(list(self.layers), segments=len(self.layers), input=h)
        else:
            for layer in self.layers:
                h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _instantiate_model(
    mode: str,
    *,
    seed: int,
    device: torch.device,
    vocab_size: int,
    d_model: int,
    layers: int,
    block_size: int,
) -> nn.Module:
    torch.manual_seed(seed)
    base = TinyGPT2(vocab_size=vocab_size, d_model=d_model, n_layer=layers)
    base_state = base.state_dict()

    if mode == "naive":
        model = TinyGPT2(vocab_size=vocab_size, d_model=d_model, n_layer=layers)
        model.load_state_dict(base_state)
        return model.to(device)

    if mode == "checkpoint":
        model = TinyGPT2(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layer=layers,
            use_checkpoint=True,
        )
        model.load_state_dict(base_state)
        return model.to(device)

    if mode == "tiny":
        inner = TinyGPT2(vocab_size=vocab_size, d_model=d_model, n_layer=layers)
        inner.load_state_dict(base_state)
        config = HCConfig(block_size=block_size)
        model = HCModule(inner, config=config)
        return model.to(device)

    raise ValueError(f"Unknown mode: {mode}")


def run_benchmark(args: argparse.Namespace) -> list[BenchmarkResult]:
    device = _select_device()
    torch.manual_seed(0)
    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

    results: list[BenchmarkResult] = []
    for mode in args.modes:
        for trial in range(args.trials):
            seed = args.seed + trial
            model = _instantiate_model(
                mode,
                seed=seed,
                device=device,
                vocab_size=args.vocab_size,
                d_model=args.d_model,
                layers=args.layers,
                block_size=args.block_size,
            )

            def forward_fn(model=model) -> torch.Tensor:
                model.zero_grad(set_to_none=True)
                logits = model(inputs)
                return torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    inputs.view(-1),
                )

            def backward_fn(loss: torch.Tensor, model=model) -> None:
                if isinstance(model, HCModule):
                    model.backward(loss)
                else:
                    loss.backward()

            result = run_single_trial(
                "gpt2",
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
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", choices=PROFILES.keys(), default="medium")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--export", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()
    args = _apply_profile(args)
    if args.batch_size is None:
        args.batch_size = 4
    if args.seq_len is None:
        args.seq_len = 128
    if args.vocab_size is None:
        args.vocab_size = 1000
    if args.d_model is None:
        args.d_model = 128
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
