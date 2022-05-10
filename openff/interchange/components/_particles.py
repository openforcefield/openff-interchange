"""
Temporary module for second-class virtual site objects.
"""
import math
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


# It is assumed that the first "orientation" atom is the "parent" atom.


class _BondChargeVirtualSite(DefaultModel):
    type: Literal["BondCharge"]
    distance: FloatQuantity["nanometer"]
    orientations: Tuple[int, ...]

    @property
    def local_frame_weights(self) -> Tuple[List[float]]:

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


class _MonovalentLonePairVirtualSite(DefaultModel):
    type: Literal["MonovalentLonePair"]
    distance: FloatQuantity["nanometer"]
    out_of_plane_angle: FloatQuantity["degree"]
    in_plane_angle: FloatQuantity["degree"]
    orientations: Tuple[int, ...]

    @property
    def local_frame_weights(self) -> Tuple[List[float]]:

        originwt = [1.0, 0.0, 0.0]
        xdir = [-1.0, 1.0, 0.0]
        ydir = [-1.0, 0.0, 1.0]

        return originwt, xdir, ydir

    @property
    def local_frame_positions(self) -> unit.Quantity:
        theta = self.in_plane_angle.m_as(unit.radian)
        phi = self.out_of_plane_angle.m_as(unit.radian)

        distance_unit = self.distance.units

        return unit.Quantity(
            [
                self.distance.m_as(distance_unit) * math.cos(theta) * math.cos(phi),
                self.distance.m_as(distance_unit) * math.sin(theta) * math.cos(phi),
                self.distance.m_as(distance_unit) * math.sin(phi),
            ],
            distance_unit,
        )


class _DivalentLonePairVirtualSite(DefaultModel):
    type: Literal["DivalentLonePair"]
    distance: FloatQuantity["nanometer"]
    out_of_plane_angle: FloatQuantity["degree"]
    orientations: Tuple[int, ...]

    @property
    def local_frame_weights(self) -> Tuple[List[float]]:

        originwt = [1.0, 0.0, 0.0]
        xdir = [-1.0, 0.5, 0.5]
        ydir = [-1.0, 1.0, 0.0]

        return originwt, xdir, ydir

    @property
    def local_frame_positions(self) -> unit.Quantity:
        theta = self.out_of_plane_angle.m_as(unit.radian)

        distance_unit = self.distance.units

        return unit.Quantity(
            [
                -self.distance.m_as(distance_unit) * math.cos(theta),
                0.0,
                self.distance.m_as(distance_unit) * math.sin(theta),
            ],
            distance_unit,
        )


class _TrivalentLonePairVirtualSite(DefaultModel):
    type: Literal["TrivalentLonePair"]
    distance: FloatQuantity["nanometer"]
    orientations: Tuple[int, ...]

    @property
    def local_frame_weights(self) -> Tuple[List[float]]:

        originwt = [1.0, 0.0, 0.0, 0.0]
        xdir = [-1.0, 1 / 3, 1 / 3, 1 / 3]
        ydir = [-1.0, 1.0, 0.0, 0.0]  # Not used

        return originwt, xdir, ydir

    @property
    def local_frame_positions(self) -> unit.Quantity:
        distance_unit = self.distance.units
        return unit.Quantity(
            [-self.distance.m_as(distance_unit), 0.0, 0.0],
            distance_unit,
        )
