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
    GROMACSVirtualSite3fad,
)
from openff.interchange.models import VirtualSiteKey
from openff.interchange.smirnoff._virtual_sites import (
    _BondChargeVirtualSite,
    _DivalentLonePairVirtualSite,
    _MonovalentLonePairVirtualSite,
    _VirtualSite,
)


def _create_gromacs_virtual_site(
    interchange: Interchange,
    virtual_site: "_VirtualSite",
    virtual_site_key: VirtualSiteKey,
    particle_map: dict[Union[int, VirtualSiteKey], int],
) -> GROMACSVirtualSite:

    # Orientation atom indices are topology indices, but here they need to be indexed as molecule
    # indices. Store the difference between an orientation atom's molecule and topology indices.
    # (It can probably be any of the orientation atoms.)
    parent_atom = interchange.topology.atom(
        virtual_site_key.orientation_atom_indices[0],
    )

    # This lookup scales poorly with system size, but it's not clear how to work around the
    # tool's ~O(N) scaling of topology lookups
    offset = interchange.topology.atom_index(
        parent_atom,
    ) - parent_atom.molecule.atom_index(parent_atom)

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

    if isinstance(virtual_site, _MonovalentLonePairVirtualSite):
        if virtual_site.out_of_plane_angle != 0.0:
            raise NotImplementedError(
                "Non-zero out-of-plane angles not yet supported in GROMACS export.",
            )

        # In the plane of three atoms, GROMACS calls this 3fad and gives the example
        #
        # [ virtual_sites3 ]
        # ; Site  from               funct   theta      d
        # 5       1     2     3      3       120        0.5
        # https://manual.gromacs.org/current/reference-manual/topologies/particle-type.html

        return GROMACSVirtualSite3fad(
            name=virtual_site_key.name,
            site=particle_map[virtual_site_key] - offset + 1,
            orientation_atoms=gromacs_indices,
            theta=virtual_site.in_plane_angle.m_as(unit.degree),
            d=virtual_site.distance.m_as(unit.nanometer),
        )

    raise NotImplementedError()
