"""Candidate-tree generation implementation."""

from .._registry import load as _load

SOURCE_MODULES = {"eagle": "specdecoding.model.ea_model"}


def load(name: str = "eagle"):
    return _load(SOURCE_MODULES, name)


__all__ = ["SOURCE_MODULES", "load"]
