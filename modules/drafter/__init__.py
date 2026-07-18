"""Drafter networks and their configuration."""

from .._registry import load as _load

SOURCE_MODULES = {
    "choices": "specdecoding.model.choices",
    "config": "specdecoding.model.configs",
    "networks": "specdecoding.model.cnets",
}


def load(name: str):
    return _load(SOURCE_MODULES, name)


__all__ = ["SOURCE_MODULES", "load"]
