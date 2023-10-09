"""
Helper functions for exporting virutal sites to OpenMM.
"""
from typing import Union

import numpy
from openff.units import Quantity, unit
from openff.units.openmm import to_openmm
from openff.utilities.utilities import has_package

from openff.interchange import Interchange
from openff.interchange.components._particles import (
    _BondChargeVirtualSite,
    _DivalentLonePairVirtualSite,
    _VirtualSite,
)
from openff.interchange.interop._virtual_sites import _get_separation_by_atom_indices
from openff.interchange.models import VirtualSiteKey

if has_package("openmm"):
    import openmm


def _create_openmm_virtual_site(
    interchange: Interchange,
    virtual_site: "_VirtualSite",
    openff_openmm_particle_map: dict[Union[int, VirtualSiteKey], int],
) -> openmm.VirtualSite:
    # virtual_site.orientations is a list of the _openff_ indices, which is more or less
    # the topology index in a topology containing only atoms (no virtual site). This dict,
    # _if only looking up atoms_, can be used to map between openff "indices" and
    # openmm "indices", where the openff "index" is the atom's index in the (openff) topology
    # and the openmm "index" is the atom's index, as a particle, in the openmm system. This
    # mapping has a different meaning if looking up a virtual site, but that should not happen here
    # as a virtual site's orientation atom should never be a virtual site
    openmm_indices: list[int] = [
        openff_openmm_particle_map[openff_index]
        for openff_index in virtual_site.orientations
    ]

    if isinstance(virtual_site, _BondChargeVirtualSite):
        separation = _get_separation_by_atom_indices(
            interchange=interchange,
            atom_indices=virtual_site.orientations,
        )
        distance = virtual_site.distance

        ratio = (distance / separation).m_as(unit.dimensionless)

        return openmm.TwoParticleAverageSite(
            *openmm_indices,
            1.0 + ratio,
            0.0 - ratio,
        )

    if isinstance(virtual_site, _DivalentLonePairVirtualSite):
        r12 = _get_separation_by_atom_indices(
            interchange=interchange,
            atom_indices=virtual_site.orientations[:2],
        )
        r13 = _get_separation_by_atom_indices(
            interchange=interchange,
            atom_indices=(virtual_site.orientations[0], virtual_site.orientations[2]),
        )

        distance = virtual_site.distance

        # TODO: Test r12 != r13, prima facia the math also applies, probably need
        #       a more direct way to get r1mid
        if r12 == r13 and float(virtual_site.out_of_plane_angle.m) == 0.0:
            r23 = _get_separation_by_atom_indices(
                interchange=interchange,
                atom_indices=virtual_site.orientations[1:],
            )

            theta = Quantity(
                numpy.arccos(
                    (r23**2 - r12**2 - r13**2) / (-2 * r12 * r13),
                ),
                unit.radian,
            )

            r1mid = Quantity(
                numpy.cos(theta.m_as(unit.radian) / 2) * r12.m_as(unit.nanometer),
                unit.nanometer,
            )

            w1 = 1 + distance / r1mid

            return openmm.ThreeParticleAverageSite(
                *openmm_indices,
                w1,
                (1 - w1) / 2,
                (1 - w1) / 2,
            )

    # It is assumed that the first "orientation" atom is the "parent" atom.
    originwt, xdir, ydir = virtual_site.local_frame_weights
    pos = virtual_site.local_frame_positions

    return openmm.LocalCoordinatesSite(
        openmm_indices,
        originwt,
        xdir,
        ydir,
        to_openmm(pos),
    )
