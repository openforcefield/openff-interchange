"""Utilities for interoperability with multiple packages."""
from typing import Union

from openff.interchange import Interchange
from openff.interchange.components._particles import _VirtualSite
from openff.interchange.exceptions import UnsupportedExportError
from openff.interchange.models import VirtualSiteKey
from openff.interchange.smirnoff import SMIRNOFFVirtualSiteCollection


def _check_virtual_site_exclusion_policy(handler: "SMIRNOFFVirtualSiteCollection"):
    _SUPPORTED_EXCLUSION_POLICIES = ("parents",)

    if handler.exclusion_policy not in _SUPPORTED_EXCLUSION_POLICIES:
        raise UnsupportedExportError(
            f"Found unsupported exclusion policy {handler.exclusion_policy}. "
            f"Supported exclusion policies are {_SUPPORTED_EXCLUSION_POLICIES}",
        )


def _build_typemap(interchange: Interchange) -> dict[int, str]:
    typemap = dict()
    elements: dict[str, int] = dict()

    # TODO: Think about how this logic relates to atom name/type clashes
    for atom_index, atom in enumerate(interchange.topology.atoms):
        element_symbol = atom.symbol
        # TODO: Use this key to condense, see parmed.openmm._process_nobonded
        # parameters = _get_lj_parameters([*parameters.values()])
        # key = tuple([*parameters.values()])

        if element_symbol not in elements.keys():
            elements[element_symbol] = 1
        else:
            elements[element_symbol] += 1

        atom_type = f"{element_symbol}{elements[element_symbol]}"
        typemap[atom_index] = atom_type

    return typemap


def _build_particle_map(
    interchange: Interchange,
    molecule_virtual_site_map,
) -> dict[Union[int, VirtualSiteKey], int]:
    particle_map: dict[Union[int, VirtualSiteKey], int] = dict()

    particle_index = 0

    for molecule in interchange.topology.molecules:
        for atom in molecule.atoms:
            atom_index = interchange.topology.atom_index(atom)

            particle_map[atom_index] = particle_index

            particle_index += 1

    for molecule in interchange.topology.molecules:
        for virtual_site_key in molecule_virtual_site_map[
            interchange.topology.molecule_index(molecule)
        ]:
            particle_map[virtual_site_key] = particle_index

            particle_index += 1

    return particle_map


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
