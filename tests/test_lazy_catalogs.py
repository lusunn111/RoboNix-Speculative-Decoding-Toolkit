from unittest.mock import patch

try:
    from speculative_decoding_service.modules import _registry
    from speculative_decoding_service.modules.acceptance import SOURCE_MODULES as ACCEPTANCE
    from speculative_decoding_service.modules.drafter import SOURCE_MODULES as DRAFTER
    from speculative_decoding_service.modules.verification import SOURCE_MODULES as VERIFICATION

    REGISTRY_IMPORT = "speculative_decoding_service.modules._registry.importlib.import_module"
except ModuleNotFoundError:
    from modules import _registry
    from modules.acceptance import SOURCE_MODULES as ACCEPTANCE
    from modules.drafter import SOURCE_MODULES as DRAFTER
    from modules.verification import SOURCE_MODULES as VERIFICATION

    REGISTRY_IMPORT = "modules._registry.importlib.import_module"


def test_pipeline_catalog_has_drafter_verifier_and_fallback():
    assert "networks" in DRAFTER
    assert "current" in VERIFICATION
    assert "kv_cache" in ACCEPTANCE


def test_registry_defers_real_model_import():
    sentinel = object()
    with patch(REGISTRY_IMPORT, return_value=sentinel):
        assert _registry.load({"mock": "mock.model"}, "mock") is sentinel
