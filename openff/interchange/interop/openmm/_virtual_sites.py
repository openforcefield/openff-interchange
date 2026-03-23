"""
Helper functions for exporting virutal sites to OpenMM.
"""

from openff.utilities import has_package

from openff.interchange import Interchange
from openff.interchange.components._particles import _VirtualSite
from openff.interchange.interop._virtual_sites import _OpenMMVirtualSite
from openff.interchange.models import BaseVirtualSiteKey

if has_package("openmm"):
    import openmm


def _create_openmm_virtual_site(
    interchange: Interchange,
    virtual_site: _VirtualSite | _OpenMMVirtualSite,
    openff_openmm_particle_map: dict[int | BaseVirtualSiteKey, int],
) -> openmm.VirtualSite:
    if isinstance(virtual_site, _VirtualSite):
        # virtual_site.orientations is a list of the _openff_ indices, which is more or less
        # the topology index in a topology containing only atoms (no virtual site). This dict,
        # _if only looking up atoms_, can be used to map between openff "indices" and
        # openmm "indices", where the openff "index" is the atom's index in the (openff) topology
        # and the openmm "index" is the atom's index, as a particle, in the openmm system. This
        # mapping has a different meaning if looking up a virtual site, but that should not happen here
        # as a virtual site's orientation atom should never be a virtual site
        openmm_indices: list[int] = [
            openff_openmm_particle_map[openff_index] for openff_index in virtual_site.orientations
        ]

        # It is assumed that the first "orientation" atom is the "parent" atom.
        originwt, xdir, ydir = virtual_site.local_frame_weights
        pos = virtual_site.local_frame_positions

        return openmm.LocalCoordinatesSite(
            openmm_indices,
            originwt,
            xdir,
            ydir,
            pos.to_openmm(),
        )
    elif isinstance(virtual_site, _OpenMMVirtualSite):
        return virtual_site.to_openmm()
