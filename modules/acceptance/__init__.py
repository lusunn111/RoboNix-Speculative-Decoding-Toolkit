"""Acceptance, fallback, and KV-cache update support."""

from .._registry import load as _load

SOURCE_MODULES = {
    "kv_cache": "specdecoding.model.kv_cache",
    "generation": "local_transformers.generation_utils",
}


def load(name: str = "kv_cache"):
    return _load(SOURCE_MODULES, name)


__all__ = ["SOURCE_MODULES", "load"]
