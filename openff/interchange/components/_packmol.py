"""
A wrapper around PACKMOL. Adapted from OpenFF Evaluator v0.4.3.
"""

import warnings

from openff.interchange.packmol import (
    RHOMBIC_DODECAHEDRON,
    RHOMBIC_DODECAHEDRON_XYHEX,
    UNIT_CUBE,
    pack_box,
    solvate_topology,
    solvate_topology_nonwater,
)
from openff.interchange.warnings import InterchangeDeprecationWarning

__all__ = (
    "RHOMBIC_DODECAHEDRON",
    "RHOMBIC_DODECAHEDRON_XYHEX",
    "UNIT_CUBE",
    "pack_box",
    "solvate_topology",
    "solvate_topology_nonwater",
)

warnings.warn(
    "This submodule is now part of the public API. Import from `openff.interchange.packmol` instead.",
    InterchangeDeprecationWarning,
)
