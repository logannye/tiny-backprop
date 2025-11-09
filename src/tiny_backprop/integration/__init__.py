"""
Framework integration entry points.

Subpackages:
- `torch`: PyTorch integration (HCModule wrapper, FX capture).
- `jax`: JAX integration (planned).
"""

from . import torch as torch_integration  # noqa: F401
from . import jax as jax_integration      # noqa: F401

__all__ = ["torch_integration", "jax_integration"]
