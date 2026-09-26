"""MCP exposure for the pinned Chemotools provider.

Importing :mod:`nir_core.chemotools_mcp` does not start a server. The stdio entry point is
``python -m nir_core.chemotools_mcp.chemotools_server`` (or the ``chemotools-mcp`` console
script when the ``nir-core[mcp]`` extra is installed).
"""

from .catalog import (
    CHEMOTOOLS_PINNED_VERSION,
    build_capability_catalog,
    describe_capability,
)

__all__ = [
    "CHEMOTOOLS_PINNED_VERSION",
    "build_capability_catalog",
    "describe_capability",
]
