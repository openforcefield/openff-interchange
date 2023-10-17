"""
Helper functions for exporting virutal sites to OpenMM.
"""
from typing import Union

from openff.units.openmm import to_openmm
from openff.utilities.utilities import has_package

from openff.interchange import Interchange
from openff.interchange.components._particles import (
    _BondChargeVirtualSite,
    _DivalentLonePairVirtualSite,
    _VirtualSite,
)
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
        from openff.interchange.interop._virtual_sites import _get_bond_charge_weights

        w1, w2 = _get_bond_charge_weights(
            virtual_site,
            interchange,
        )

        return openmm.TwoParticleAverageSite(
            *openmm_indices,
            w1,
            w2,
        )

    if isinstance(virtual_site, _DivalentLonePairVirtualSite):
        from openff.interchange.interop._virtual_sites import _get_divalent_weights

        if virtual_site.out_of_plane_angle.m == 0.0:
            w1, w2, w3 = _get_divalent_weights(
                virtual_site,
                interchange,
            )

            return openmm.ThreeParticleAverageSite(
                *openmm_indices,
                w1,
                w2,
                w3,
            )

        else:
            w12, w13, wcross = _get_divalent_weights(
                virtual_site,
                interchange,
            )

            return openmm.OutOfPlaneSite(
                *openmm_indices,
                w12,
                w13,
                wcross,
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
