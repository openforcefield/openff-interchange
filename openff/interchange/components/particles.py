"""
Temporary module for second-class virtual site objects.
"""
from typing import List, Literal, Tuple

from openff.units import unit
from openff.units.openmm import to_openmm
from openmm import LocalCoordinatesSite

from openff.interchange.models import DefaultModel
from openff.interchange.types import FloatQuantity


def _create_openmm_virtual_site(virtual_site, atoms):
    originwt, xdir, ydir = virtual_site.local_frame_weights
    pos = virtual_site.local_frame_positions
    return LocalCoordinatesSite(atoms, originwt, xdir, ydir, to_openmm(pos))


class _BondChargeVirtualSite(DefaultModel):
    type: Literal["BondCharge"]
    distance: FloatQuantity["nanometer"]
    orientations: Tuple[int, ...]

    # It is assumed that the first "orientation" atom is the "parent" atom.

    @property
    def local_frame_weights(self) -> Tuple[List[int]]:
        originwt = [1.0, 0.0]  # first atom is origin
        xdir = [-1.0, 1.0]
        ydir = [-1.0, 1.0]

        return originwt, xdir, ydir

    @property
    def local_frame_positions(self) -> unit.Quantity:
        distance_unit = self.distance.units
        return unit.Quantity(
            [-self.distance.m_as(distance_unit), 0.0, 0.0],
            distance_unit,
        )
