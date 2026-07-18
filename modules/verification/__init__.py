"""Candidate verification and posterior evaluation implementations."""

from .._registry import load as _load

SOURCE_MODULES = {
    "current": "specdecoding.model.utils_c",
    "original": "specdecoding.model.utils_origin",
    "other": "specdecoding.model.utils_others",
}


def load(name: str = "current"):
    return _load(SOURCE_MODULES, name)


__all__ = ["SOURCE_MODULES", "load"]
