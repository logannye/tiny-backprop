"""
PyTorch integration for tiny-backprop.

Exports:
- `HCModule`: nn.Module wrapper that will use a height-compressed schedule.
"""

from .wrapped_module import HCConfig, HCModule

__all__ = ["HCModule", "HCConfig"]
