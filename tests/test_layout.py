import importlib.util
from pathlib import Path
import subprocess
import sys

try:
    import speculative_decoding_service as service
    from speculative_decoding_service.modules import strategies

    SERVICE_ROOT = Path(service.__file__).resolve().parent
    activate_vendor = service.activate_vendor
except ModuleNotFoundError:
    import service_bootstrap
    from modules import strategies

    SERVICE_ROOT = Path(service_bootstrap.__file__).resolve().parent
    activate_vendor = service_bootstrap.activate_vendor


def test_vendor_tree_and_default_strategy_exist():
    assert (SERVICE_ROOT / "vendor" / "openvla" / "specdecoding" / "model").is_dir()
    assert strategies.DEFAULT_STRATEGY in strategies.STRATEGIES


def test_activate_vendor_is_idempotent():
    assert activate_vendor() == activate_vendor()


def test_flattened_vendor_exposes_openvla_namespace():
    activate_vendor()
    assert importlib.util.find_spec("openvla.prismatic") is not None
    assert importlib.util.find_spec("openvla.specdecoding") is not None


def test_cli_help_works_from_independent_toolkit_root():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run", "--help"],
        cwd=SERVICE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Path relative to vendor/openvla" in result.stdout
