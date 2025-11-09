from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch  # type: ignore
from torch import nn  # type: ignore

from tiny_backprop.integration.torch.fx_capture import capture_graph
from tiny_backprop.integration.torch.wrapped_module import HCConfig, HCModule


class TinyChain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, x):
        return self.layers(x)


def test_fx_capture_chain_graph() -> None:
    module = TinyChain()
    example = torch.randn(2, 8)

    graph = capture_graph(module, example_inputs=example)

    assert graph.metadata["framework"] == "torch_fx"
    graph.validate()

    topo = graph.topological_sort()
    assert len(topo) == len(graph.nodes)
    assert graph.inputs == [topo[0].name]
    assert graph.outputs
    assert all(node.outputs_size >= 0 for node in topo)


def test_hc_module_backward_matches_autograd() -> None:
    torch.manual_seed(0)
    base_model = TinyChain()
    reference_model = TinyChain()
    reference_model.load_state_dict(base_model.state_dict())

    x = torch.randn(2, 8, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    baseline = reference_model(x_ref).sum()
    baseline.backward()
    baseline_grad = x_ref.grad.detach().clone()

    hc_model = HCModule(base_model, config=HCConfig(block_size=2))
    out = hc_model(x).sum()
    hc_model.backward(out)

    assert torch.allclose(x.grad, baseline_grad, atol=1e-5)
    assert hc_model.executor is not None
    assert hc_model.executor.checkpoint_plan is not None


def test_hc_module_fallback_when_capture_fails(monkeypatch) -> None:
    torch.manual_seed(0)

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

    def boom(*args, **kwargs):
        raise RuntimeError("capture failure")

    hc_model = HCModule(Tiny(), config=HCConfig(block_size=2))
    monkeypatch.setattr(
        "tiny_backprop.integration.torch.wrapped_module.capture_graph", boom
    )

    x = torch.randn(2, 4, requires_grad=True)
    loss = hc_model(x).sum()
    hc_model.backward(loss)

    assert hc_model.executor is not None
    plan = hc_model.executor.checkpoint_plan
    assert plan is not None
    assert plan.save_nodes == []

