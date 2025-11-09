"""
Minimal PyTorch examples using HCModule.

These are smoke tests / reference usage, not benchmark-quality scripts.
"""

from __future__ import annotations

import torch
from torch import nn

from .wrapped_module import HCConfig, HCModule


def simple_mlp_demo() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 4),
    )
    hc_model = HCModule(model, config=HCConfig(block_size=4))

    x = torch.randn(8, 16)
    y = hc_model(x).sum()
    hc_model.backward(y)

    executor = hc_model.executor
    if executor and executor.checkpoint_plan:
        print(
            "Plan blocks:",
            executor.checkpoint_plan.meta.get("num_blocks"),
            "Block size:",
            executor.checkpoint_plan.meta.get("block_size"),
        )
    print("simple_mlp_demo: backward() completed with height-compressed planning.")


if __name__ == "__main__":
    simple_mlp_demo()
