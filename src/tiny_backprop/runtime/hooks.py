"""
Hooks for capturing activations and mapping them to graph nodes.

Framework-specific code will wrap these.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable


class ActivationRegistry:
    """
    Very simple activation store; real impl will be tensor-type aware.
    """

    def __init__(self) -> None:
        self._store: Dict[int, Any] = {}

    def save(self, key: int, value: Any) -> None:
        self._store[key] = value

    def load(self, key: int) -> Any:
        return self._store[key]

    def delete(self, key: int) -> None:
        self._store.pop(key, None)

    def has(self, key: int) -> bool:
        return key in self._store

    def clear(self) -> None:
        self._store.clear()

    def items(self) -> Iterable[tuple[int, Any]]:
        return self._store.items()
