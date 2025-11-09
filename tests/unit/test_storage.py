from __future__ import annotations

import pytest

from tiny_backprop.runtime.storage import CheckpointStorage


def test_checkpoint_storage_put_get_delete() -> None:
    storage = CheckpointStorage()
    storage.put("a", 1)
    assert storage.get("a") == 1

    storage.put("a", 2)
    assert storage.get("a") == 2

    storage.delete("a")
    with pytest.raises(KeyError):
        _ = storage.get("a")


def test_checkpoint_storage_has_and_clear() -> None:
    storage = CheckpointStorage()
    storage.put("x", 42)
    assert storage.has("x")
    storage.clear()
    assert not storage.has("x")

