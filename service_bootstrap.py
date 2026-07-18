"""Runtime helpers for loading the import-compatible KERV source snapshot."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any, Dict

SERVICE_ROOT = Path(__file__).resolve().parent
VENDOR_OPENVLA_ROOT = SERVICE_ROOT / "vendor" / "openvla"


def activate_vendor() -> Path:
    """Expose the vendored OpenVLA tree without importing GPU dependencies."""
    value = str(VENDOR_OPENVLA_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)
    return VENDOR_OPENVLA_ROOT


def run_vendor_script(relative_path: str) -> Dict[str, Any]:
    """Run an original KERV script with its CLI arguments unchanged."""
    root = activate_vendor()
    script = (root / relative_path).resolve()
    if root.resolve() not in script.parents or not script.is_file():
        raise FileNotFoundError(f"Unknown KERV script: {relative_path}")
    return runpy.run_path(str(script), run_name="__main__")
