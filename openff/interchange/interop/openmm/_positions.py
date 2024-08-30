"""
Helper functions for exporting positions to OpenMM.
"""

from typing import TYPE_CHECKING

from openff.interchange.exceptions import (
    MissingPositionsError,
    MissingVirtualSitesError,
)

if TYPE_CHECKING:
    import openmm.unit

    from openff.interchange import Interchange


def to_openmm_positions(
    interchange: "Interchange",
    include_virtual_sites: bool = True,
) -> "openmm.unit.Quantity":
    """Generate an array of positions of all particles, optionally including virtual sites."""
    if interchange.positions is None:
        raise MissingPositionsError(
            f"Positions are required, found {interchange.positions=}.",
        )

    if include_virtual_sites:
        from openff.interchange.interop._virtual_sites import (
            get_positions_with_virtual_sites,
        )

        try:
            return get_positions_with_virtual_sites(
                interchange,
                use_zeros=False,
            ).to_openmm()
        except MissingVirtualSitesError:
            return interchange.positions.to_openmm()
    else:
        return interchange.positions.to_openmm()
