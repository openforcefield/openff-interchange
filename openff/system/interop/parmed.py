from typing import Any

import numpy as np
import parmed as pmd

from .. import unit


def to_parmed(off_system: Any) -> pmd.Structure:
    """Convert an OpenFF System to a ParmEd Structure"""
    structure = pmd.Structure()
    _convert_box(off_system, structure)

    for topology_molecule in off_system.topology.topology_molecules:
        for atom in topology_molecule.atoms:
            structure.add_atom(
                pmd.Atom(atomic_number=atom.atomic_number), resname="", resnum=0
            )

    for bond in off_system.topology.topology_bonds:
        atom1, atom2 = bond.atoms
        structure.bonds.append(
            pmd.Bond(atom1.toplogy_atom_index, atom2.topology_atom_index)
        )

    # TODO: How to populate angles & dihedrals?

    vdw_term = off_system.term_collection.terms["vdW"]
    for pmd_idx, pmd_atom in enumerate(structure.atoms):
        potential = vdw_term.potentials[vdw_term.smirks_map[(pmd_idx,)]]
        sigma, epsilon = _lj_params_from_potential(potential)
        pmd_atom.sigma = sigma
        pmd_atom.epsilon = epsilon

    electrostatics_term = off_system.term_collection.terms["Electrostatics"]
    for pmd_idx, pmd_atom in enumerate(structure.atoms):
        partial_charge = (
            electrostatics_term.potentials[str(pmd_idx)]
            .to(unit.elementary_charge)
            .magnitude
        )
        pmd_atom.charge = partial_charge

    structure.positions = off_system.positions.to(unit.angstrom).m

    return structure


def _convert_box(off_system: Any, structure: pmd.Structure) -> None:
    # TODO: Convert box vectors to box lengths + angles
    lengths = off_system.box.to(unit("angstrom")).diagonal().magnitude
    angles = 3 * [90]
    structure.box = np.hstack([lengths, angles])


def _lj_params_from_potential(potential):
    sigma = potential.parameters["sigma"].to(unit.angstrom)
    epsilon = potential.parameters["epsilon"].to(unit.Unit("kilocalorie/mol"))

    return sigma.magnitude, epsilon.magnitude
