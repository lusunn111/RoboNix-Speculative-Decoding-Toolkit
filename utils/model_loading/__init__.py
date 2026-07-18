"""Model-loading paths retained in vendor/openvla/prismatic/models."""

try:
    from speculative_decoding_service.modules._registry import load as _load
except ModuleNotFoundError as exc:
    if exc.name != "speculative_decoding_service":
        raise
    from modules._registry import load as _load

SOURCE_MODULES = {"loader": "prismatic.models.load", "materialize": "prismatic.models.materialize"}


def load(name: str = "loader"):
    return _load(SOURCE_MODULES, name)


__all__ = ["SOURCE_MODULES", "load"]
