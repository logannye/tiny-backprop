from __future__ import annotations

from tiny_backprop.runtime.hooks import ActivationRegistry


def test_activation_registry_save_load_delete() -> None:
    registry = ActivationRegistry()
    registry.save(1, "value")
    assert registry.load(1) == "value"
    assert registry.has(1)

    registry.delete(1)
    assert not registry.has(1)


def test_activation_registry_clear() -> None:
    registry = ActivationRegistry()
    registry.save(1, "a")
    registry.save(2, "b")
    registry.clear()
    assert not list(registry.items())

