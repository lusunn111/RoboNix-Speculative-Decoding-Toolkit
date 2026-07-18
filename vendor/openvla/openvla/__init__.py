"""Compatibility namespace for the flattened OpenVLA source snapshot.

The upstream project stores ``prismatic``, ``specdecoding``, and
``experiments`` below an ``openvla/`` package directory.  The organized
snapshot keeps those trees at ``vendor/openvla/`` so they remain visible to
the stable runner.  Extending this package's search path to the parent restores
imports such as ``openvla.prismatic`` without copying the large source tree.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent)]

