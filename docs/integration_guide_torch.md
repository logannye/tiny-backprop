# PyTorch Integration Guide

`tiny-backprop` provides a drop-in wrapper (`HCModule`) that plans a
height-compressed schedule while staying API-compatible with standard `nn.Module`
interfaces.

## Installation

```bash
pip install -e .
pip install torch torchvision  # choose CUDA / CPU build as needed
```

Ensure your environment has GPU drivers if you plan to measure CUDA memory usage.

## Basic Usage

```python
import torch
from torch import nn
from tiny_backprop.integration.torch import HCConfig, HCModule

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        return self.net(x)

model = HCModule(ToyModel(), config=HCConfig(block_size=16))
x = torch.randn(32, 128)
loss = model(x).mean()
model.backward(loss)

executor = model.executor
print("Blocks:", executor.checkpoint_plan.meta["num_blocks"])
```

When `block_size` is omitted, the executor defaults to `ceil(sqrt(num_nodes))`.

## Planning Lifecycle

1. The first forward pass captures an FX graph and stores example inputs.
2. `configure_executor` (called lazily by `backward`) runs analysis, builds a
   checkpoint plan, and stores the executor.
3. Subsequent backward calls reuse the cached executor unless you reset it (e.g. if
   the model’s structure changes).

If you need manual control:

```python
executor = model.configure_executor(example_inputs=(torch.randn(32, 128),))
# inspect executor.checkpoint_plan.meta here
model.backward(loss)
```

## Custom Planners

Provide a callable that returns a configured `HeightCompressedExecutor`:

```python
from tiny_backprop.runtime.executor import HeightCompressedExecutor

def my_planner(module, example_inputs):
    graph = capture_graph(module, example_inputs)
    executor = HeightCompressedExecutor()
    executor.configure(graph, block_size=8)
    return executor

model = HCModule(ToyModel(), config=HCConfig(planner=my_planner))
```

Your planner can incorporate bespoke heuristics, lower-bound checks, or offline
profiling before returning the executor.

## Working with Callbacks

`HCModule.backward` accepts an optional `callbacks` argument that maps to
`ExecutionCallbacks`. This unlocks custom recompute/backward orchestration (e.g.,
to interleave with mixed-precision or pipeline-parallel engines).

```python
from tiny_backprop.runtime.executor import ExecutionCallbacks

events = []

def run_forward(start, end):
    events.append(("forward", start, end))

def run_backward(start, end):
    events.append(("backward", start, end))

callbacks = ExecutionCallbacks(run_forward=run_forward, run_backward=run_backward)
model.backward(loss, callbacks=callbacks)
```

If callbacks are omitted, the executor simply calls `torch.autograd.backward`.

## Troubleshooting

- **Capture failures**: the wrapper falls back to the naive executor when FX cannot
  symbolically trace the module (e.g., control flow depending on tensor data). Consider
  using `torch.fx.proxy` tools or manual graph construction in such cases.
- **Changing model topology**: if you mutate `self.module` (e.g., swapping layers),
  call `model.configure_executor(example_inputs=...)` again to rebuild the plan.
- **Memory estimates**: checkpoint plans rely on tensor metadata from `ShapeProp`;
  ensure that your inputs are representative so activation sizes are computed
  accurately.

## Benchmarks

Use the scripts under `benchmarks/` for structured comparisons:

```bash
python -m benchmarks.mem_vs_time_transformer --modes naive checkpoint tiny --trials 3
python -m benchmarks.mem_vs_time_resnet --export results/resnet.json
```

These scripts print summary tables and optionally emit JSON for dashboarding.

