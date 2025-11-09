from __future__ import annotations

from tiny_backprop.graph.ir import Graph, Node
from tiny_backprop.runtime.executor import ExecutionCallbacks, HeightCompressedExecutor


def _build_chain_graph(num_nodes: int) -> Graph:
    graph = Graph()
    prev = None
    for idx in range(num_nodes):
        name = f"n{idx}"
        inputs = [prev] if prev is not None else []
        graph.add_node(Node(name=name, op="linear", inputs=inputs, outputs_size=idx + 1))
        if prev is None:
            graph.inputs = [name]
        prev = name
    graph.outputs = [prev] if prev is not None else []
    graph.validate()
    return graph


def test_executor_configure_builds_interval_plan() -> None:
    graph = _build_chain_graph(4)
    executor = HeightCompressedExecutor()
    executor.configure(graph)  # default block size

    assert executor.checkpoint_plan is not None
    assert executor.replay_plan is not None
    assert executor.replay_plan.steps  # non-empty schedule


def test_executor_backward_with_callbacks_runs_schedule() -> None:
    graph = _build_chain_graph(4)
    executor = HeightCompressedExecutor()
    executor.configure(graph, block_size=2)

    assert executor.replay_plan is not None
    expected_steps = [
        (step.kind, step.start_idx, step.end_idx) for step in executor.replay_plan
    ]

    events: list[tuple[str, int, int]] = []
    loss_holder = []
    finalized = []

    def run_forward(start: int, end: int) -> None:
        events.append(("forward", start, end))

    def run_backward(start: int, end: int) -> None:
        events.append(("backward", start, end))

    def prepare_loss(loss) -> None:
        loss_holder.append(loss)

    def finalize() -> None:
        finalized.append(True)

    callbacks = ExecutionCallbacks(
        run_forward=run_forward,
        run_backward=run_backward,
        prepare_loss=prepare_loss,
        finalize=finalize,
    )

    loss = object()
    autograd_called = {"flag": False}

    def autograd_backward(received_loss) -> None:
        autograd_called["flag"] = True
        loss_holder.append(received_loss)

    executor.backward(loss, autograd_backward, callbacks=callbacks)

    assert not autograd_called["flag"]
    assert loss_holder and loss_holder[0] is loss
    assert finalized

    assert events == expected_steps


def test_executor_backward_without_callbacks_falls_back_to_autograd() -> None:
    graph = _build_chain_graph(3)
    executor = HeightCompressedExecutor()
    executor.configure(graph, block_size=0)  # force naive plan

    loss = object()
    autograd_called = {"flag": False}

    def autograd_backward(received_loss) -> None:
        autograd_called["flag"] = True
        assert received_loss is loss

    executor.backward(loss, autograd_backward, callbacks=None)
    assert autograd_called["flag"]

