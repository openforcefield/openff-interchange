"""
Helper functions for exporting virutal sites to OpenMM.
"""
from collections.abc import Iterable
from typing import Union

from openff.units import Quantity, unit
from openff.units.openmm import to_openmm
from openff.utilities.utilities import has_package

from openff.interchange import Interchange
from openff.interchange.components._particles import (
    _BondChargeVirtualSite,
    _VirtualSite,
)
from openff.interchange.exceptions import UnsupportedExportError
from openff.interchange.models import VirtualSiteKey
from openff.interchange.smirnoff._virtual_sites import SMIRNOFFVirtualSiteCollection

if has_package("openmm"):
    import openmm


def _check_virtual_site_exclusion_policy(handler: "SMIRNOFFVirtualSiteCollection"):
    _SUPPORTED_EXCLUSION_POLICIES = ("parents",)

    if handler.exclusion_policy not in _SUPPORTED_EXCLUSION_POLICIES:
        raise UnsupportedExportError(
            f"Found unsupported exclusion policy {handler.exclusion_policy}. "
            f"Supported exclusion policies are {_SUPPORTED_EXCLUSION_POLICIES}",
        )


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
            virtual_site.orientations[0],
            virtual_site.orientations[1],
            1.0 + ratio,
            0.0 - ratio,
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


def _get_separation_by_atom_indices(
    interchange: Interchange,
    atom_indices: Iterable[int],
) -> Quantity:
    """
    Given indices of (two?) atoms, return the distance between them.

    A constraint distance is first searched for, then an equilibrium bond length.

    This is slow, but often necessary for converting virtual site "distances" to weighted
    averages (unitless) of orientation atom positions.
    """
    if "Constraints" in interchange.collections:
        collection = interchange["Constraints"]

        for key in collection.key_map:
            if (key.atom_indices == atom_indices) or (
                key.atom_indices[::-1] == atom_indices
            ):
                return collection.potentials[collection.key_map[key]].parameters[
                    "distance"
                ]

    if "Bonds" in interchange.collections:
        collection = interchange["Bonds"]

        for key in collection.key_map:
            if (key.atom_indices == atom_indices) or (
                key.atom_indices[::-1] == atom_indices
            ):
                return collection.potentials[collection.key_map[key]].parameters[
                    "length"
                ]

    raise ValueError(f"Could not find distance between atoms {atom_indices}")
