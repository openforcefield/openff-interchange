import re
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
                pmd.Atom(atomic_number=atom.atomic_number), resname="FOO", resnum=0
            )

    if "Bonds" in off_system.term_collection.terms:
        bond_term = off_system.term_collection.terms["Bonds"]
        for bond, smirks in bond_term.smirks_map.items():
            idx_1, idx_2 = bond
            pot = bond_term.potentials[bond_term.smirks_map[bond]]
            k = pot.parameters["k"].m / 2
            length = pot.parameters["length"].m
            bond_type = pmd.BondType(k=k, req=length)
            structure.bonds.append(
                pmd.Bond(
                    atom1=structure.atoms[idx_1],
                    atom2=structure.atoms[idx_2],
                    type=bond_type,
                )
            )

    if "Angles" in off_system.term_collection.terms:
        angle_term = off_system.term_collection.terms["Angles"]
        for angle, smirks in angle_term.smirks_map.items():
            idx_1, idx_2, idx_3 = angle
            pot = angle_term.potentials[angle_term.smirks_map[angle]]
            # TODO: Look at cost of redundant conversions, to ensure correct units of .m
            k = pot.parameters["k"].magnitude  # kcal/mol/rad**2
            theta = pot.parameters["angle"].magnitude  # degree
            angle_type = pmd.AngleType(k=k, theteq=theta)
            structure.angles.append(
                pmd.Angle(
                    atom1=structure.atoms[idx_1],
                    atom2=structure.atoms[idx_2],
                    atom3=structure.atoms[idx_3],
                    type=angle_type,
                )
            )

    if "ProperTorsions" in off_system.term_collection.terms:
        proper_term = off_system.term_collection.terms["ProperTorsions"]
        for proper, smirks in proper_term.smirks_map.items():
            idx_1, idx_2, idx_3, idx_4 = proper
            pot = proper_term.potentials[proper_term.smirks_map[proper]]
            # TODO: Better way of storing periodic data in generally, probably need to improve Potential
            n = re.search(r"\d", "".join(pot.parameters.keys())).group()
            k = pot.parameters["k" + n].m  # kcal/mol
            periodicity = pot.parameters["periodicity" + n].m  # dimless
            phase = pot.parameters["phase" + n].m  # degree

            dihedral_type = pmd.DihedralType(per=periodicity, phi_k=k, phase=phase)
            structure.dihedrals.append(
                pmd.Dihedral(
                    atom1=structure.atoms[idx_1],
                    atom2=structure.atoms[idx_2],
                    atom3=structure.atoms[idx_3],
                    atom4=structure.atoms[idx_4],
                    type=dihedral_type,
                )
            )

    if "ImroperTorsions" in off_system.term_collection.terms:
        improper_term = off_system.term_collection.terms["ImproperTorsions"]
        for improper, smirks in improper_term.smirks_map.items():
            idx_1, idx_2, idx_3, idx_4 = improper
            pot = improper_term.potentials[improper_term.smirks_map[improper]]
            # TODO: Better way of storing periodic data in generally, probably need to improve Potential
            n = re.search(r"\d", "".join(pot.parameters.keys())).group()
            k = pot.parameters["k" + n].m  # kcal/mol
            periodicity = pot.parameters["periodicity" + n].m  # dimless
            phase = pot.parameters["phase" + n].m  # degree

            dihedral_type = pmd.DihedralType(per=periodicity, phi_k=k, phase=phase)
            structure.dihedrals.append(
                pmd.Dihedral(
                    atom1=structure.atoms[idx_1],
                    atom2=structure.atoms[idx_2],
                    atom3=structure.atoms[idx_3],
                    atom4=structure.atoms[idx_4],
                    type=dihedral_type,
                )
            )

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

    # Assign dummy residue names, GROMACS will not accept empty strings
    for res in structure.residues:
        res.name = "FOO"

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
