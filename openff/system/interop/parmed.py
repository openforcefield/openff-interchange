from typing import TYPE_CHECKING, Dict

import numpy as np
import parmed as pmd

from openff.system import unit
from openff.system.components.potentials import Potential
from openff.system.models import PotentialKey, TopologyKey

if TYPE_CHECKING:
    from openff.system.components.system import System


kcal_mol = unit.Unit("kilocalories / mol")
kcal_mol_a2 = unit.Unit("kilocalories / mol / angstrom ** 2")
kcal_mol_rad2 = unit.Unit("kilocalories / mol / rad ** 2")


def to_parmed(off_system: "System") -> pmd.Structure:
    """Convert an OpenFF System to a ParmEd Structure"""
    structure = pmd.Structure()
    _convert_box(off_system.box, structure)

    if "Electrostatics" in off_system.handlers.keys():
        has_electrostatics = True
        electrostatics_handler = off_system.handlers["Electrostatics"]
    else:
        has_electrostatics = False

    for topology_molecule in off_system.topology.topology_molecules:  # type: ignore[union-attr]
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
        bond_map: Dict = dict()
        for top_key, pot_key in bond_handler.slot_map.items():
            idx_1, idx_2 = top_key.atom_indices
            try:
                bond_type = bond_map[pot_key]
            except KeyError:
                pot = bond_handler.potentials[pot_key]
                k = pot.parameters["k"].to(kcal_mol_a2).magnitude / 2
                length = pot.parameters["length"].to(unit.angstrom).magnitude
                bond_type = pmd.BondType(k=k, req=length)
                bond_map[pot_key] = bond_type
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
            del bond_type, idx_1, idx_2, pot_key, top_key

    if "Angles" in off_system.handlers.keys():
        angle_term = off_system.handlers["Angles"]
        for top_key, pot_key in angle_term.slot_map.items():
            idx_1, idx_2, idx_3 = top_key.atom_indices
            pot = angle_term.potentials[pot_key]
            # TODO: Look at cost of redundant conversions, to ensure correct units of .m
            k = pot.parameters["k"].to(kcal_mol_rad2).magnitude / 2
            theta = pot.parameters["angle"].to(unit.degree).magnitude
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
    vdw_14 = off_system.handlers["vdW"].scale_14  # type: ignore[attr-defined]
    if has_electrostatics:
        coul_14 = off_system.handlers["Electrostatics"].scale_14  # type: ignore[attr-defined]
    else:
        coul_14 = 1.0
    vdw_handler = off_system.handlers["vdW"]
    if "ProperTorsions" in off_system.handlers.keys():
        proper_term = off_system.handlers["ProperTorsions"]
        for top_key, pot_key in proper_term.slot_map.items():
            idx_1, idx_2, idx_3, idx_4 = top_key.atom_indices
            pot = proper_term.potentials[pot_key]
            for n in range(pot.parameters["n_terms"]):
                k = pot.parameters["k"][n].to(kcal_mol).magnitude
                periodicity = pot.parameters["periodicity"][n]
                phase = pot.parameters["phase"][n].magnitude
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
                key1 = TopologyKey(atom_indices=(idx_1,))
                key4 = TopologyKey(atom_indices=(idx_4,))
                vdw1 = vdw_handler.potentials[vdw_handler.slot_map[key1]]
                vdw4 = vdw_handler.potentials[vdw_handler.slot_map[key4]]
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
        top_key = TopologyKey(atom_indices=(pmd_idx,))
        smirks = vdw_handler.slot_map[top_key]
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
            top_key = TopologyKey(atom_indices=(pmd_idx,))
            partial_charge = electrostatics_handler.charges[top_key]  # type: ignore[attr-defined]
            unitless_ = partial_charge.to(unit.elementary_charge).magnitude
            pmd_atom.charge = float(unitless_)
            pmd_atom.atom_type.charge = float(unitless_)
        else:
            pmd_atom.charge = 0

    # Assign dummy residue names, GROMACS will not accept empty strings
    for res in structure.residues:
        res.name = "FOO"

    structure.positions = off_system.positions.to(unit.angstrom).magnitude  # type: ignore[attr-defined]
    for idx, pos in enumerate(structure.positions):
        structure.atoms[idx].xx = pos._value[0]
        structure.atoms[idx].xy = pos._value[1]
        structure.atoms[idx].xz = pos._value[2]

    return structure


def from_parmed(cls) -> "System":

    from openff.system.components.system import System

    out = System()

    if cls.positions:
        out.positions = np.asarray(cls.positions._value) * unit.angstrom

    if any(cls.box[3:] != 3 * [90.0]):
        from openff.system.exceptions import UnsupportedBoxError

        raise UnsupportedBoxError(
            f"Found box with angles {cls.box[3:]}. Only"
            "rectangular boxes are currently supported."
        )

    out.box = cls.box[:3] * unit.angstrom

    from openff.toolkit.topology import Molecule, Topology

    top = Topology()

    for res in cls.residues:
        mol = Molecule()
        mol.name = res.name
        for atom in res.atoms:
            mol.add_atom(
                atomic_number=atom.atomic_number, formal_charge=0, is_aromatic=False
            )
            for bond in atom.bonds:
                try:
                    mol.add_bond(
                        atom1=bond.atom1,
                        atom2=bond.atom1,
                        bond_order=bond.order,
                    )
                # TODO: Use a custom exception after
                # https://github.com/openforcefield/openff-toolkit/issues/771
                except Exception:
                    pass

        top.add_molecule(mol)

    out.topology = top

    from openff.system.components.smirnoff import (
        ElectrostaticsMetaHandler,
        SMIRNOFFBondHandler,
        SMIRNOFFvdWHandler,
    )

    vdw_handler = SMIRNOFFvdWHandler()
    coul_handler = ElectrostaticsMetaHandler()

    for atom in cls.atoms:
        atom_idx = atom.idx
        sigma = atom.sigma * unit.angstrom
        epsilon = atom.epsilon * kcal_mol
        charge = atom.charge * unit.elementary_charge
        top_key = TopologyKey(atom_indices=(atom_idx,))
        pot_key = PotentialKey(id=str(atom_idx))
        pot = Potential(parameters={"sigma": sigma, "epsilon": epsilon})

        vdw_handler.slot_map.update({top_key: pot_key})
        vdw_handler.potentials.update({pot_key: pot})

        coul_handler.charges.update({top_key: charge})

    bond_handler = SMIRNOFFBondHandler()

    for bond in cls.bonds:
        atom1 = bond.atom1
        atom2 = bond.atom2
        k = bond.type.k * kcal_mol_a2
        length = bond.type.req * unit.angstrom
        top_key = TopologyKey(atom_indices=(atom1.idx, atom2.idx))
        pot_key = PotentialKey(id=f"{atom1.idx}-{atom2.idx}")
        pot = Potential(parameters={"k": k, "length": length})

        bond_handler.slot_map.update({top_key: pot_key})
        bond_handler.potentials.update({pot_key: pot})

    out.handlers.update({"vdW": vdw_handler})
    out.handlers.update({"Electrostatics": coul_handler})  # type: ignore[dict-item]
    out.handlers.update({"Bonds": bond_handler})

    return out


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
    sigma = potential.parameters["sigma"].to(unit.angstrom).magnitude
    epsilon = potential.parameters["epsilon"].to(kcal_mol).magnitude

    return sigma, epsilon
