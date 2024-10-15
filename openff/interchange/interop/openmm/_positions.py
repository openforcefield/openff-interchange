"""
Helper functions for exporting positions to OpenMM.
"""

from typing import TYPE_CHECKING

from openff.interchange.interop.common import _to_positions

if TYPE_CHECKING:
    import openmm.unit

    from openff.interchange import Interchange


def to_openmm_positions(
    interchange: "Interchange",
    include_virtual_sites: bool = True,
) -> "openmm.unit.Quantity":
    """Generate an array of positions of all particles, optionally excluding virtual sites."""
    return _to_positions(interchange, include_virtual_sites).to_openmm()
