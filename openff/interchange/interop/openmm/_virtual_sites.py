from typing import TYPE_CHECKING

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
) -> openmm.LocalCoordinatesSite:

    # It is assumed that the first "orientation" atom is the "parent" atom.
    originwt, xdir, ydir = virtual_site.local_frame_weights
    pos = virtual_site.local_frame_positions
    return openmm.LocalCoordinatesSite(
        virtual_site.orientations, originwt, xdir, ydir, to_openmm(pos)
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
    # parent_atom = orientations[0]

    if virtual_site_key.type == "BondCharge":
        virtual_site_object = _BondChargeVirtualSite(
            type="BondCharge",
            distance=virtual_site_potential.parameters["distance"],
            orientations=orientations,
        )
    elif virtual_site_key.type == "MonovalentLonePair":
        virtual_site_object = _MonovalentLonePairVirtualSite(
            type="MonovalentLonePair",
            distance=virtual_site_potential.parameters["distance"],
            out_of_plane_angle=virtual_site_potential.parameters["outOfPlaneAngle"],
            in_plane_angle=virtual_site_potential.parameters["inPlaneAngle"],
            orientations=orientations,
        )
    elif virtual_site_key.type == "DivalentLonePair":
        virtual_site_object = _DivalentLonePairVirtualSite(
            type="DivalentLonePair",
            distance=virtual_site_potential.parameters["distance"],
            out_of_plane_angle=virtual_site_potential.parameters["outOfPlaneAngle"],
            orientations=orientations,
        )
    elif virtual_site_key.type == "TrivalentLonePair":
        virtual_site_object = _TrivalentLonePairVirtualSite(
            type="TrivalentLonePair",
            distance=virtual_site_potential.parameters["distance"],
            orientations=orientations,
        )

    else:
        raise NotImplementedError(virtual_site_key.type)

    return virtual_site_object
