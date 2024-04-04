"""Utilities for interoperability with multiple packages."""

from openff.interchange import Interchange
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
    collate: bool = False,
) -> dict[int | VirtualSiteKey, int]:
    """
    Build a dict mapping particle indices between a topology and another object.

    If `collate=True`, virtual sites are collated with each molecule's atoms.
    If `collate=False`, virtual sites go at the very end, after all atoms were added.
    """
    particle_map: dict[int | VirtualSiteKey, int] = dict()

    particle_index = 0

    for molecule in interchange.topology.molecules:
        for atom in molecule.atoms:
            atom_index = interchange.topology.atom_index(atom)

            particle_map[atom_index] = particle_index

            particle_index += 1

        if collate:
            for virtual_site_key in molecule_virtual_site_map[
                interchange.topology.molecule_index(molecule)
            ]:
                particle_map[virtual_site_key] = particle_index

                particle_index += 1

    if not collate:
        for molecule in interchange.topology.molecules:
            for virtual_site_key in molecule_virtual_site_map[
                interchange.topology.molecule_index(molecule)
            ]:
                particle_map[virtual_site_key] = particle_index

                particle_index += 1

    return particle_map
