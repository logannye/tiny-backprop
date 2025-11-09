"""
Abstract storage backend for checkpoints (GPU/CPU/offload).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable


class CheckpointStorage:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def put(self, name: str, tensor: Any) -> None:
        self._store[name] = tensor

    def get(self, name: str) -> Any:
        return self._store[name]

    def delete(self, name: str) -> None:
        self._store.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._store

    def clear(self) -> None:
        self._store.clear()

    def items(self) -> Iterable[tuple[str, Any]]:
        return self._store.items()
