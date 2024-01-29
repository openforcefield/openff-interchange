"""
Helper functions for exporting virutal sites to GROMACS.
"""

from typing import Union

import numpy
from openff.units import Quantity, unit

from openff.interchange import Interchange
from openff.interchange.interop._virtual_sites import _get_separation_by_atom_indices
from openff.interchange.interop.gromacs.models.models import (
    GROMACSVirtualSite,
    GROMACSVirtualSite2,
    GROMACSVirtualSite3,
)
from openff.interchange.models import VirtualSiteKey
from openff.interchange.smirnoff._virtual_sites import (
    _BondChargeVirtualSite,
    _DivalentLonePairVirtualSite,
    _VirtualSite,
)


def _create_gromacs_virtual_site(
    interchange: Interchange,
    virtual_site: "_VirtualSite",
    virtual_site_key: VirtualSiteKey,
    particle_map: dict[Union[int, VirtualSiteKey], int],
) -> GROMACSVirtualSite:
    offset = interchange.topology.atom_index(
        interchange.topology.atom(min(virtual_site_key.orientation_atom_indices)),
    )

    # These are GROMACS "molecule" indices, already mapped back from the topology on to the molecule
    gromacs_indices: list[int] = [
        particle_map[openff_index] - offset + 1
        for openff_index in virtual_site.orientations
    ]

    if isinstance(virtual_site, _BondChargeVirtualSite):
        separation = _get_separation_by_atom_indices(
            interchange=interchange,
            atom_indices=virtual_site.orientations,
        )
        distance = virtual_site.distance

        ratio = (distance / separation).m_as(unit.dimensionless)

        # TODO: This will create name collisions in many molecules
        return GROMACSVirtualSite2(
            name=virtual_site_key.name,
            site=particle_map[virtual_site_key] - offset + 1,
            orientation_atoms=gromacs_indices,
            a=-1.0 * ratio,  # this is basically w2 in OpenMM jargon
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

            return GROMACSVirtualSite3(
                name=virtual_site_key.name,
                site=particle_map[virtual_site_key] - offset + 1,
                orientation_atoms=gromacs_indices,
                a=(1 - w1) / 2,
                b=(1 - w1) / 2,
            )

        else:
            raise NotImplementedError()

    raise NotImplementedError()
