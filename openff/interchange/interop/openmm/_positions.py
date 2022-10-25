from typing import TYPE_CHECKING

from openff.units import unit as off_unit
from openff.units.openmm import to_openmm as to_openmm_quantity

from openff.interchange.exceptions import MissingPositionsError

if TYPE_CHECKING:
    from openff.interchange import Interchange


def to_openmm_positions(
    interchange: "Interchange",
    include_virtual_sites: bool = True,
) -> off_unit.Quantity:
    """Generate an array of positions of all particles, optionally including virtual sites."""
    from collections import defaultdict

    import numpy

    if interchange.positions is None:
        raise MissingPositionsError(
            f"Positions are required found {interchange.positions=}."
        )

    atom_positions = to_openmm_quantity(interchange.positions)

    if "VirtualSites" not in interchange.handlers:
        return atom_positions
    elif len(interchange["VirtualSites"].slot_map) == 0:
        return atom_positions

    topology = interchange.topology

    if include_virtual_sites:
        from openff.interchange.interop._virtual_sites import (
            _virtual_site_parent_molecule_mapping,
        )

        virtual_site_molecule_map = _virtual_site_parent_molecule_mapping(interchange)

        molecule_virtual_site_map = defaultdict(list)

        for virtual_site, molecule_index in virtual_site_molecule_map.items():
            molecule_virtual_site_map[molecule_index].append(virtual_site)

    particle_positions = off_unit.Quantity(
        numpy.empty(shape=(0, 3)), off_unit.nanometer
    )

    for molecule in topology.molecules:
        molecule_index = topology.molecule_index(molecule)

        try:
            this_molecule_atom_positions = molecule.conformers[0]
        except TypeError:
            atom_indices = [topology.atom_index(atom) for atom in molecule.atoms]
            this_molecule_atom_positions = interchange.positions[atom_indices, :]
            # Interchange.position is populated, but Molecule.conformers is not

        if include_virtual_sites:
            n_virtual_sites_in_this_molecule: int = len(
                molecule_virtual_site_map[molecule_index]
            )
            this_molecule_virtual_site_positions = off_unit.Quantity(
                numpy.zeros((n_virtual_sites_in_this_molecule, 3)), off_unit.nanometer
            )
            particle_positions = numpy.concatenate(
                [
                    particle_positions,
                    this_molecule_atom_positions,
                    this_molecule_virtual_site_positions,
                ]
            )

        else:
            particle_positions = numpy.concatenate(
                [
                    particle_positions,
                    this_molecule_atom_positions,
                ]
            )

    return particle_positions
