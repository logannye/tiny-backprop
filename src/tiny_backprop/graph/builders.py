"""
Builders from framework-specific graphs (PyTorch FX, JAXPR, etc.) into Graph.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

from .ir import Graph, Node


def from_pytorch_fx(
    fx_graph_module: Any,
    *,
    example_inputs: Optional[Sequence[Any]] = None,
    populate_tensor_meta: bool = True,
) -> Graph:
    """
    Convert a ``torch.fx.GraphModule`` into a ``Graph``.

    Args:
        fx_graph_module: Result of ``torch.fx.symbolic_trace`` or similar.
        example_inputs: Optional inputs used to run ``ShapeProp`` so that we
            can extract tensor metadata and estimate activation sizes.
        populate_tensor_meta: If ``True`` (default), attempts to store tensor
            metadata in node attrs for later analysis.
    """
    try:
        import torch.fx as fx
        from torch.fx.graph_module import GraphModule
        from torch.fx.passes.shape_prop import ShapeProp
    except ModuleNotFoundError as exc:  # pragma: no cover - handled in tests
        raise ModuleNotFoundError(
            "from_pytorch_fx requires PyTorch to be installed."
        ) from exc

    if not isinstance(fx_graph_module, GraphModule):
        raise TypeError(
            "from_pytorch_fx expects a torch.fx.GraphModule. "
            f"Received: {type(fx_graph_module)!r}"
        )

    gm: GraphModule = fx_graph_module

    if example_inputs is not None:
        if not isinstance(example_inputs, (list, tuple)):
            example_inputs = (example_inputs,)
        gm.eval()
        ShapeProp(gm).propagate(*example_inputs)

    graph = Graph(metadata={"framework": "torch_fx"})

    def _sanitize(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            return [_sanitize(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _sanitize(v) for k, v in value.items()}
        return repr(value)

    def _tensor_meta_size(meta: Any) -> int:
        if meta is None:
            return 0
        if isinstance(meta, (list, tuple)):
            return sum(_tensor_meta_size(m) for m in meta)
        shape = getattr(meta, "shape", None)
        dtype = getattr(meta, "dtype", None)
        if shape is None or dtype is None:
            return 0
        try:
            itemsize = _torch_dtype_size(dtype)
        except Exception:
            return 0
        numel = 1
        for dim in shape:
            if dim is None or dim < 0:
                return 0
            numel *= dim
        return int(numel * itemsize)

    def _pack_tensor_meta(meta: Any) -> Any:
        if meta is None:
            return None
        if isinstance(meta, (list, tuple)):
            return [_pack_tensor_meta(m) for m in meta]
        shape = getattr(meta, "shape", None)
        dtype = getattr(meta, "dtype", None)
        if shape is None or dtype is None:
            return repr(meta)
        return {
            "shape": tuple(int(dim) if dim is not None else -1 for dim in shape),
            "dtype": str(dtype),
            "requires_grad": bool(getattr(meta, "requires_grad", False)),
        }

    def _node_activation_size(node: fx.Node) -> int:
        if example_inputs is None:
            return 0
        return _tensor_meta_size(node.meta.get("tensor_meta"))

    def _collect_output_nodes(node: fx.Node) -> List[str]:
        names: List[str] = []

        def visit(arg: Any) -> None:
            if isinstance(arg, fx.Node):
                names.append(arg.name)
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    visit(item)
            elif isinstance(arg, dict):
                for item in arg.values():
                    visit(item)

        for arg in node.args:
            visit(arg)
        return names

    for fx_node in gm.graph.nodes:
        if fx_node.op == "output":
            graph.outputs = _collect_output_nodes(fx_node)
            continue

        inputs = [input_node.name for input_node in fx_node.all_input_nodes]
        attrs: Dict[str, Any] = {
            "fx_op": fx_node.op,
            "target": _sanitize(fx_node.target),
            "kwargs": _sanitize(fx_node.kwargs),
        }
        if populate_tensor_meta:
            attrs["tensor_meta"] = _pack_tensor_meta(fx_node.meta.get("tensor_meta"))

        node = Node(
            name=fx_node.name,
            op=str(fx_node.op),
            inputs=inputs,
            outputs_size=_node_activation_size(fx_node),
            attrs=attrs,
        )

        graph.add_node(node)

        if fx_node.op == "placeholder":
            graph.inputs.append(fx_node.name)

    graph.validate()
    return graph


def from_jaxpr(jaxpr: Any) -> Graph:
    """
    Convert a ``jax.core.ClosedJaxpr`` (or ``Jaxpr``) into a ``Graph``.
    """
    try:
        import numpy as np
        try:
            from jax import core as jax_core
        except ModuleNotFoundError:
            from jax._src import core as jax_core  # type: ignore[attr-defined]
        else:
            if not hasattr(jax_core, "ClosedJaxpr"):
                from jax._src import core as jax_core  # type: ignore[attr-defined]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "from_jaxpr requires JAX to be installed."
        ) from exc

    ClosedJaxpr = getattr(jax_core, "ClosedJaxpr", None)
    Jaxpr = getattr(jax_core, "Jaxpr", None)
    Literal = getattr(jax_core, "Literal", None)
    Var = getattr(jax_core, "Var", None)
    AbstractValue = getattr(jax_core, "AbstractValue", None)

    if ClosedJaxpr is None or Jaxpr is None or Var is None:
        raise RuntimeError("Unsupported JAX version: core types not available.")

    closed_jaxpr: ClosedJaxpr
    if isinstance(jaxpr, ClosedJaxpr):
        closed_jaxpr = jaxpr
    elif isinstance(jaxpr, Jaxpr):
        closed_jaxpr = ClosedJaxpr(jaxpr, consts=[])
    else:
        raise TypeError(
            "from_jaxpr expects a jax.core.ClosedJaxpr or Jaxpr. "
            f"Received: {type(jaxpr)!r}"
        )

    inner = closed_jaxpr.jaxpr
    const_vals = list(closed_jaxpr.consts)

    graph = Graph(metadata={"framework": "jaxpr"})

    def _freeze(obj: Any) -> Any:
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        if isinstance(obj, (list, tuple)):
            return [_freeze(o) for o in obj]
        if isinstance(obj, dict):
            return {str(k): _freeze(v) for k, v in obj.items()}
        return repr(obj)

    def _aval_size(aval: AbstractValue) -> int:
        if not hasattr(aval, "shape") or not hasattr(aval, "dtype"):
            return 0
        shape = getattr(aval, "shape")
        try:
            numel = int(math.prod(int(dim) for dim in shape))
        except TypeError:
            return 0
        try:
            itemsize = np.dtype(getattr(aval, "dtype")).itemsize
        except TypeError:
            return 0
        return int(numel * itemsize)

    def _value_size(value: Any) -> int:
        if value is None:
            return 0
        if hasattr(value, "size") and hasattr(value, "dtype"):
            try:
                return int(value.size) * np.dtype(value.dtype).itemsize
            except TypeError:
                return 0
        if isinstance(value, (int, float, bool)):
            return value.__sizeof__()
        return 0

    var_to_node: Dict[Var, str] = {}

    for idx, (var, const_val) in enumerate(zip(inner.constvars, const_vals)):
        name = f"const_{idx}"
        graph.add_node(
            Node(
                name=name,
                op="const",
                inputs=[],
                outputs_size=_aval_size(var.aval) or _value_size(const_val),
                attrs={
                    "aval": _freeze(getattr(var, "aval", None)),
                    "value": _freeze(const_val),
                },
            )
        )
        var_to_node[var] = name

    for pos, var in enumerate(inner.invars):
        if Literal is not None and isinstance(var, Literal):
            name = f"literal_input_{pos}"
            graph.add_node(
                Node(
                    name=name,
                    op="literal",
                    inputs=[],
                    outputs_size=_value_size(var.val),
                    attrs={"value": _freeze(var.val)},
                )
            )
            graph.inputs.append(name)
            continue

        node_name = f"input_{pos}"
        graph.add_node(
            Node(
                name=node_name,
                op="input",
                inputs=[],
                outputs_size=_aval_size(var.aval),
                attrs={"var": str(var), "aval": _freeze(var.aval)},
            )
        )
        graph.inputs.append(node_name)
        var_to_node[var] = node_name

    for idx, eqn in enumerate(inner.eqns):
        input_nodes: List[str] = []
        for invar in eqn.invars:
            if Literal is not None and isinstance(invar, Literal):
                continue
            producer = var_to_node.get(invar)
            if producer is None:
                raise KeyError(
                    f"JAXPR contains var `{invar}` with no registered producer."
                )
            input_nodes.append(producer)

        node_name = f"{eqn.primitive.name}_{idx}"
        outputs_size = sum(
            _aval_size(outvar.aval)
            for outvar in eqn.outvars
            if isinstance(outvar, Var)
        )

        graph.add_node(
            Node(
                name=node_name,
                op=str(eqn.primitive.name),
                inputs=sorted(set(input_nodes)),
                outputs_size=outputs_size,
                attrs={
                    "params": _freeze(eqn.params),
                    "invars": [str(v) for v in eqn.invars],
                    "outvars": [str(v) for v in eqn.outvars],
                },
            )
        )

        for outvar in eqn.outvars:
            if isinstance(outvar, Var):
                var_to_node[outvar] = node_name

    outputs: List[str] = []
    for pos, outvar in enumerate(inner.outvars):
        if Literal is not None and isinstance(outvar, Literal):
            name = f"literal_output_{pos}"
            graph.add_node(
                Node(
                    name=name,
                    op="literal",
                    inputs=[],
                    outputs_size=_value_size(outvar.val),
                    attrs={"value": _freeze(outvar.val)},
                ),
                allow_overwrite=True,
            )
            outputs.append(name)
            continue

        producer = var_to_node.get(outvar)
        if producer is None:
            raise KeyError(
                f"Output variable `{outvar}` has no registered producer in graph."
            )
        outputs.append(producer)

    graph.outputs = outputs
    graph.validate()
    return graph


@lru_cache(maxsize=None)
def _torch_dtype_size(dtype: Any) -> int:
    import torch

    return int(torch.tensor(0, dtype=dtype).element_size())
