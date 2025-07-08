"""
A wrapper around PACKMOL. Adapted from OpenFF Evaluator v0.4.3.
"""

from openff.interchange.packmol import (
    RHOMBIC_DODECAHEDRON,
    RHOMBIC_DODECAHEDRON_XYHEX,
    UNIT_CUBE,
    pack_box,
    solvate_topology,
    solvate_topology_nonwater,
)

__all__ = (
    "RHOMBIC_DODECAHEDRON",
    "RHOMBIC_DODECAHEDRON_XYHEX",
    "UNIT_CUBE",
    "pack_box",
    "solvate_topology",
    "solvate_topology_nonwater",
)

import warnings

warnings.warn(
    "This submodule is now part of the public API. Import from `openff.interchange.packmol instead.",
)
