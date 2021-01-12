from typing import Any

import numpy as np
import parmed as pmd

from openff.system import unit

kcal_mol = unit.Unit("kilocalories / mol")


def to_parmed(off_system: Any) -> pmd.Structure:
    """Convert an OpenFF System to a ParmEd Structure"""
    structure = pmd.Structure()
    _convert_box(off_system.box, structure)

    if "Electrostatics" in off_system.handlers.keys():
        has_electrostatics = True
        electrostatics_handler = off_system.handlers["Electrostatics"]
    else:
        has_electrostatics = False

    for topology_molecule in off_system.topology.topology_molecules:
        for atom in topology_molecule.atoms:
            atomic_number = atom.atomic_number
            element = pmd.periodic_table.Element[atomic_number]
            mass = pmd.periodic_table.Mass[element]
            structure.add_atom(
                pmd.Atom(
                    atomic_number=atomic_number,
                    mass=mass,
                ),
                resname="FOO",
                resnum=0,
            )

    if "Bonds" in off_system.handlers.keys():
        bond_handler = off_system.handlers["Bonds"]
        bond_map = dict()
        for bond_slot, smirks in bond_handler.slot_map.items():
            idx_1, idx_2 = eval(bond_slot)
            try:
                bond_type = bond_map[smirks]
            except KeyError:
                pot = bond_handler.potentials[smirks]
                k = pot.parameters["k"] / 2  # .m
                length = pot.parameters["length"]  # .m
                bond_type = pmd.BondType(k=k, req=length)
                bond_map[smirks] = bond_type
                del pot, k, length
            if bond_type not in structure.bond_types:
                structure.bond_types.append(bond_type)
            structure.bonds.append(
                pmd.Bond(
                    atom1=structure.atoms[idx_1],
                    atom2=structure.atoms[idx_2],
                    type=bond_type,
                )
            )
            del bond_type, idx_1, idx_2, smirks, bond_slot

    if "Angles" in off_system.handlers.keys():
        angle_term = off_system.handlers["Angles"]
        for angle, smirks in angle_term.slot_map.items():
            idx_1, idx_2, idx_3 = eval(angle)
            pot = angle_term.potentials[smirks]
            # TODO: Look at cost of redundant conversions, to ensure correct units of .m
            k = pot.parameters["k"] / 2  # .magnitude, kcal/mol/rad**2
            theta = pot.parameters["angle"]  # .magnitude, degree
            # TODO: Look up if AngleType already exists in struct
            angle_type = pmd.AngleType(k=k, theteq=theta)
            structure.angles.append(
                pmd.Angle(
                    atom1=structure.atoms[idx_1],
                    atom2=structure.atoms[idx_2],
                    atom3=structure.atoms[idx_3],
                    type=angle_type,
                )
            )
            structure.angle_types.append(angle_type)

    # ParmEd treats 1-4 scaling factors at the level of each DihedralType,
    # whereas SMIRNOFF captures them at the level of the non-bonded handler,
    # so they need to be stored here for processing dihedrals
    vdw_14 = off_system.handlers["vdW"].scale_14
    if has_electrostatics:
        coul_14 = off_system.handlers["Electrostatics"].scale_14
    else:
        coul_14 = 1.0
    vdw_handler = off_system.handlers["vdW"]
    if "ProperTorsions" in off_system.handlers.keys():
        proper_term = off_system.handlers["ProperTorsions"]
        for proper, smirks in proper_term.slot_map.items():
            idx_1, idx_2, idx_3, idx_4 = eval(proper)
            pot = proper_term.potentials[smirks]
            for n in range(pot.parameters["n_terms"]):
                k = pot.parameters["k"][n]  # .to(kcal_mol).magnitude
                periodicity = pot.parameters["periodicity"][n]
                phase = pot.parameters["phase"][n]  # .magnitude
                dihedral_type = pmd.DihedralType(
                    phi_k=k,
                    per=periodicity,
                    phase=phase,
                    scnb=1 / vdw_14,
                    scee=1 / coul_14,
                )
                structure.dihedrals.append(
                    pmd.Dihedral(
                        atom1=structure.atoms[idx_1],
                        atom2=structure.atoms[idx_2],
                        atom3=structure.atoms[idx_3],
                        atom4=structure.atoms[idx_4],
                        type=dihedral_type,
                    )
                )
                structure.dihedral_types.append(dihedral_type)
                vdw1 = vdw_handler.potentials[vdw_handler.slot_map[str((idx_1,))]]
                vdw4 = vdw_handler.potentials[vdw_handler.slot_map[str((idx_4,))]]
                sig1, eps1 = _lj_params_from_potential(vdw1)
                sig4, eps4 = _lj_params_from_potential(vdw4)
                sig = (sig1 + sig4) * 0.5
                eps = (eps1 * eps4) ** 0.5
                nbtype = pmd.NonbondedExceptionType(
                    rmin=sig * 2 ** (1 / 6), epsilon=eps * vdw_14, chgscale=coul_14
                )
                structure.adjusts.append(
                    pmd.NonbondedException(
                        structure.atoms[idx_1], structure.atoms[idx_4], type=nbtype
                    )
                )
                structure.adjust_types.append(nbtype)

    #    if False:  # "ImroperTorsions" in off_system.term_collection.terms:
    #        improper_term = off_system.term_collection.terms["ImproperTorsions"]
    #        for improper, smirks in improper_term.smirks_map.items():
    #            idx_1, idx_2, idx_3, idx_4 = improper
    #            pot = improper_term.potentials[improper_term.smirks_map[improper]]
    #            # TODO: Better way of storing periodic data in generally, probably need to improve Potential
    #            n = re.search(r"\d", "".join(pot.parameters.keys())).group()
    #            k = pot.parameters["k" + n].m  # kcal/mol
    #            periodicity = pot.parameters["periodicity" + n].m  # dimless
    #            phase = pot.parameters["phase" + n].m  # degree
    #
    #            dihedral_type = pmd.DihedralType(per=periodicity, phi_k=k, phase=phase)
    #            structure.dihedrals.append(
    #                pmd.Dihedral(
    #                    atom1=structure.atoms[idx_1],
    #                    atom2=structure.atoms[idx_2],
    #                    atom3=structure.atoms[idx_3],
    #                    atom4=structure.atoms[idx_4],
    #                    type=dihedral_type,
    #                )
    #            )

    vdw_handler = off_system.handlers["vdW"]
    for pmd_idx, pmd_atom in enumerate(structure.atoms):
        smirks = vdw_handler.slot_map[str((pmd_idx,))]
        potential = vdw_handler.potentials[smirks]
        element = pmd.periodic_table.Element[pmd_atom.element]
        sigma, epsilon = _lj_params_from_potential(potential)

        atom_type = pmd.AtomType(
            name=element + str(pmd_idx + 1),
            number=pmd_idx,
            atomic_number=pmd_atom.atomic_number,
            mass=pmd.periodic_table.Mass[element],
        )

        atom_type.set_lj_params(eps=epsilon, rmin=sigma * 2 ** (1 / 6) / 2)
        pmd_atom.atom_type = atom_type
        pmd_atom.type = atom_type.name
        pmd_atom.name = pmd_atom.type

    for pmd_idx, pmd_atom in enumerate(structure.atoms):
        if has_electrostatics:
            partial_charge = electrostatics_handler.charge_map[str((pmd_idx,))]
            partial_charge_unitless = (
                partial_charge  # .to(unit.elementary_charge).magnitude
            )
            pmd_atom.charge = float(partial_charge_unitless)
            pmd_atom.atom_type.charge = float(partial_charge_unitless)
        else:
            pmd_atom.charge = 0

    # Assign dummy residue names, GROMACS will not accept empty strings
    for res in structure.residues:
        res.name = "FOO"

    structure.positions = off_system.positions.to(unit.angstrom).magnitude

    return structure


# def _convert_box(box: unit.Quantity, structure: pmd.Structure) -> None:
def _convert_box(box, structure: pmd.Structure) -> None:
    # TODO: Convert box vectors to box lengths + angles
    if box is None:
        lengths = [0, 0, 0]
    else:
        # TODO: Handle non-rectangular boxes
        lengths = box.diagonal().to(unit("angstrom")).magnitude
    angles = 3 * [90]
    structure.box = np.hstack([lengths, angles])


def _lj_params_from_potential(potential):
    sigma = potential.parameters["sigma"]  # .to(unit.angstrom).magnitude
    epsilon = potential.parameters["epsilon"]  # .to(kcal_mol).magnitude

    return sigma, epsilon
