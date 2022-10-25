from typing import TYPE_CHECKING, Dict, List, Union

import openmm
from openff.units.openmm import to_openmm

from openff.interchange.exceptions import UnsupportedExportError
from openff.interchange.models import VirtualSiteKey

if TYPE_CHECKING:
    from openff.interchange.components._particles import _VirtualSite
    from openff.interchange.components.smirnoff import SMIRNOFFVirtualSiteHandler


def _check_virtual_site_exclusion_policy(handler: "SMIRNOFFVirtualSiteHandler"):
    _SUPPORTED_EXCLUSION_POLICIES = ("parents",)

    if handler.exclusion_policy not in _SUPPORTED_EXCLUSION_POLICIES:
        raise UnsupportedExportError(
            f"Found unsupported exclusion policy {handler.exclusion_policy}. "
            f"Supported exclusion policies are {_SUPPORTED_EXCLUSION_POLICIES}"
        )


def _create_openmm_virtual_site(
    virtual_site: "_VirtualSite",
    openff_openmm_particle_map: Dict[Union[int, VirtualSiteKey], int],
) -> openmm.LocalCoordinatesSite:

    # It is assumed that the first "orientation" atom is the "parent" atom.
    originwt, xdir, ydir = virtual_site.local_frame_weights  # type: ignore[misc]
    pos = virtual_site.local_frame_positions

    # virtual_site.orientations is a list of the _openff_ indices, which is more or less
    # the topology index in a topology containing only atoms (no virtual site). This dict,
    # _if only looking up atoms_, can be used to map between openff "indices" and
    # openmm "indices", where the openff "index" is the atom's index in the (openff) topology
    # and the openmm "index" is the atom's index, as a particle, in the openmm system. This
    # mapping has a different meaning if looking up a virtual site, but that should not happen here
    # as a virtual site's orientation atom should never be a virtual site
    openmm_indices: List[int] = [
        openff_openmm_particle_map[openff_index]
        for openff_index in virtual_site.orientations
    ]

    return openmm.LocalCoordinatesSite(
        openmm_indices, originwt, xdir, ydir, to_openmm(pos)  # type: ignore[has-type]
    )


def _create_virtual_site_object(
    virtual_site_key: VirtualSiteKey,
    virtual_site_potential,
    # interchange: "Interchange",
    # non_bonded_force: openmm.NonbondedForce,
) -> "_VirtualSite":
    from openff.interchange.components._particles import (
        _BondChargeVirtualSite,
        _DivalentLonePairVirtualSite,
        _MonovalentLonePairVirtualSite,
        _TrivalentLonePairVirtualSite,
    )

    orientations = virtual_site_key.orientation_atom_indices

    if virtual_site_key.type == "BondCharge":
        return _BondChargeVirtualSite(
            type="BondCharge",
            distance=virtual_site_potential.parameters["distance"],
            orientations=orientations,
        )
    elif virtual_site_key.type == "MonovalentLonePair":
        return _MonovalentLonePairVirtualSite(
            type="MonovalentLonePair",
            distance=virtual_site_potential.parameters["distance"],
            out_of_plane_angle=virtual_site_potential.parameters["outOfPlaneAngle"],
            in_plane_angle=virtual_site_potential.parameters["inPlaneAngle"],
            orientations=orientations,
        )
    elif virtual_site_key.type == "DivalentLonePair":
        return _DivalentLonePairVirtualSite(
            type="DivalentLonePair",
            distance=virtual_site_potential.parameters["distance"],
            out_of_plane_angle=virtual_site_potential.parameters["outOfPlaneAngle"],
            orientations=orientations,
        )
    elif virtual_site_key.type == "TrivalentLonePair":
        return _TrivalentLonePairVirtualSite(
            type="TrivalentLonePair",
            distance=virtual_site_potential.parameters["distance"],
            orientations=orientations,
        )

    else:
        raise NotImplementedError(virtual_site_key.type)
