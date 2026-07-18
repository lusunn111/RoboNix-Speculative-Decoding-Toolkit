"""Lazy access to the canonical KERV KV-cache implementation."""

try:
    from speculative_decoding_service.modules.acceptance import load
except ModuleNotFoundError as exc:
    if exc.name != "speculative_decoding_service":
        raise
    from modules.acceptance import load

__all__ = ["load"]
