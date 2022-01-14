"""Interfaces with ParmEd."""
from typing import TYPE_CHECKING, Dict, Optional, Union

import mdtraj as md
import numpy as np
from openff.units import unit

from openff.interchange.components.potentials import Potential
from openff.interchange.exceptions import (
    ConversionError,
    UnsupportedBoxError,
    UnsupportedExportError,
)
from openff.interchange.models import PotentialKey, TopologyKey

if TYPE_CHECKING:

    import parmed as pmd

    from openff.interchange import Interchange
    from openff.interchange.components.smirnoff import (
        SMIRNOFFImproperTorsionHandler,
        SMIRNOFFProperTorsionHandler,
    )

kcal_mol = unit.Unit("kilocalories / mol")
kcal_mol_a2 = unit.Unit("kilocalories / mol / angstrom ** 2")
kcal_mol_rad2 = unit.Unit("kilocalories / mol / rad ** 2")


def _to_parmed(off_system: "Interchange") -> "pmd.Structure":
    """Convert an OpenFF Interchange to a ParmEd Structure."""
    import parmed as pmd

    structure = pmd.Structure()
    _convert_box(off_system.box, structure)

    if "Electrostatics" in off_system.handlers.keys():
        has_electrostatics = True
        electrostatics_handler = off_system.handlers["Electrostatics"]
    else:
        has_electrostatics = False

    for atom in off_system.topology.atoms:
        atomic_number = atom.element.atomic_number
        mass = atom.element.mass
        try:
            resname = atom.metadata["residue_number"]
            resnum = atom.metadata["residue_name"]
        except KeyError:
            resname = "UNK"
            resnum = 0

        structure.add_atom(
            pmd.Atom(
                atomic_number=atomic_number,
                mass=mass,
            ),
            resname,
            resnum,
        )

    if "Bonds" in off_system.handlers.keys():
        bond_handler = off_system.handlers["Bonds"]
        bond_type_map: Dict = dict()
        for pot_key, pot in bond_handler.potentials.items():
            k = pot.parameters["k"].to(kcal_mol_a2).magnitude / 2
            length = pot.parameters["length"].to(unit.angstrom).magnitude
            bond_type = pmd.BondType(k=k, req=length)
            bond_type_map[pot_key] = bond_type
            structure.bond_types.append(bond_type)

        for top_key, pot_key in bond_handler.slot_map.items():
            idx_1, idx_2 = top_key.atom_indices
            bond_type = bond_type_map[pot_key]
            bond = pmd.Bond(
                atom1=structure.atoms[idx_1],
                atom2=structure.atoms[idx_2],
                type=bond_type,
            )
            structure.bonds.append(bond)

    structure.bond_types.claim()

    if "Angles" in off_system.handlers.keys():
        angle_handler = off_system.handlers["Angles"]
        angle_type_map: Dict = dict()
        for pot_key, pot in angle_handler.potentials.items():
            k = pot.parameters["k"].to(kcal_mol_rad2).magnitude / 2
            theta = pot.parameters["angle"].to(unit.degree).magnitude
            # TODO: Look up if AngleType already exists in struct
            angle_type = pmd.AngleType(k=k, theteq=theta)
            angle_type_map[pot_key] = angle_type
            structure.angle_types.append(angle_type)

        for top_key, pot_key in angle_handler.slot_map.items():
            idx_1, idx_2, idx_3 = top_key.atom_indices
            angle_type = angle_type_map[pot_key]
            structure.angles.append(
                pmd.Angle(
                    atom1=structure.atoms[idx_1],
                    atom2=structure.atoms[idx_2],
                    atom3=structure.atoms[idx_3],
                    type=angle_type,
                )
            )
            structure.angle_types.append(angle_type)

    structure.angle_types.claim()

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
        proper_torsion_handler = off_system.handlers["ProperTorsions"]
        proper_type_map: Dict = dict()
        for pot_key, pot in proper_torsion_handler.potentials.items():
            k = pot.parameters["k"].to(kcal_mol).magnitude
            periodicity = pot.parameters["periodicity"]
            phase = pot.parameters["phase"].magnitude
            proper_type = pmd.DihedralType(
                phi_k=k,
                per=periodicity,
                phase=phase,
                scnb=1 / vdw_14,
                scee=1 / coul_14,
            )
            proper_type_map[pot_key] = proper_type
            structure.dihedral_types.append(proper_type)

        for top_key, pot_key in proper_torsion_handler.slot_map.items():
            idx_1, idx_2, idx_3, idx_4 = top_key.atom_indices
            dihedral_type = proper_type_map[pot_key]
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
            sig = sig.m_as(unit.angstrom)
            eps = eps.m_as(kcal_mol)
            nbtype = pmd.NonbondedExceptionType(
                rmin=sig * 2 ** (1 / 6), epsilon=eps * vdw_14, chgscale=coul_14
            )
            structure.adjusts.append(
                pmd.NonbondedException(
                    structure.atoms[idx_1], structure.atoms[idx_4], type=nbtype
                )
            )
            structure.adjust_types.append(nbtype)

    structure.dihedral_types.claim()
    structure.adjust_types.claim()

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
    if vdw_handler.mixing_rule == "lorentz-berthelot":
        structure.combining_rule = "lorentz"
    elif vdw_handler.mixing_rule == "geometric":
        structure.combining_rule = "geometric"
    else:
        raise UnsupportedExportError(
            f"ParmEd likely does not support mixing rule {vdw_handler.mixing_rule}"
        )

    for pmd_idx, pmd_atom in enumerate(structure.atoms):
        top_key = TopologyKey(atom_indices=(pmd_idx,))
        smirks = vdw_handler.slot_map[top_key]
        potential = vdw_handler.potentials[smirks]
        element = pmd.periodic_table.Element[pmd_atom.element]
        sigma, epsilon = _lj_params_from_potential(potential)
        sigma = sigma.m_as(unit.angstrom)
        epsilon = epsilon.m_as(kcal_mol)

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

    if has_electrostatics:
        charges = electrostatics_handler.charges

    for pmd_idx, pmd_atom in enumerate(structure.atoms):
        if has_electrostatics:
            top_key = TopologyKey(atom_indices=(pmd_idx,))
            partial_charge = charges[top_key]
            unitless_ = partial_charge.to(unit.elementary_charge).magnitude
            pmd_atom.charge = float(unitless_)
            pmd_atom.atom_type.charge = float(unitless_)
        else:
            pmd_atom.charge = 0

    # Assign dummy residue names, GROMACS will not accept empty strings
    for res in structure.residues:
        res.name = "FOO"

    if off_system.positions is not None:
        structure.positions = off_system.positions.to(unit.angstrom).magnitude

        for idx, pos in enumerate(structure.positions):
            structure.atoms[idx].xx = pos._value[0]
            structure.atoms[idx].xy = pos._value[1]
            structure.atoms[idx].xz = pos._value[2]

    return structure


def _from_parmed(cls, structure) -> "Interchange":
    import parmed as pmd

    out = cls()

    if structure.positions:
        out.positions = np.asarray(structure.positions._value) * unit.angstrom

    if structure.box is not None:
        if any(structure.box[3:] != 3 * [90.0]):
            raise UnsupportedBoxError(
                f"Found box with angles {structure.box[3:]}. Only"
                "rectangular boxes are currently supported."
            )

        out.box = structure.box[:3] * unit.angstrom

    if structure.topology is not None:
        from openff.toolkit.topology.topology import Topology

        mdtop = md.Topology.from_openmm(value=structure.topology)
        out.topology = Topology()
        out.topology.mdtop = mdtop
    else:
        raise ConversionError("ParmEd Structure missing an topology attribute")

    from openff.interchange.components.smirnoff import (
        SMIRNOFFAngleHandler,
        SMIRNOFFBondHandler,
        SMIRNOFFElectrostaticsHandler,
        SMIRNOFFImproperTorsionHandler,
        SMIRNOFFProperTorsionHandler,
        SMIRNOFFvdWHandler,
    )

    _scale_14_coul = {1 / d.scee for d in structure.dihedral_types if d.scee != 0}
    if len(_scale_14_coul) == 0:
        scale_14_coul = 0.83333
    elif len(_scale_14_coul) > 1:
        raise ConversionError(
            "Found multiple values of the 1-4 scaling factor for electrostatics"
        )
    else:
        scale_14_coul = [*_scale_14_coul][0]

    _scale_14_vdw = {1 / d.scnb for d in structure.dihedral_types if d.scnb != 0}
    if len(_scale_14_vdw) == 0:
        scale_14_vdw = 0.5
    elif len(_scale_14_vdw) > 1:
        raise ConversionError("Found multiple values of the 1-4 scaling factor for vdw")
    else:
        scale_14_vdw = [*_scale_14_vdw][0]

    # TODO: Infer 1-4 scaling factors from exceptions/adjusts/dihedrals/something
    vdw_handler = SMIRNOFFvdWHandler(scale_14=scale_14_vdw)
    coul_handler = SMIRNOFFElectrostaticsHandler(scale_14=scale_14_coul, method="pme")

    for atom in structure.atoms:
        atom_idx = atom.idx
        sigma = atom.sigma * unit.angstrom
        epsilon = atom.epsilon * kcal_mol
        charge = atom.charge * unit.elementary_charge
        top_key = TopologyKey(atom_indices=(atom_idx,))
        pot_key = PotentialKey(id=str(atom_idx))
        pot = Potential(parameters={"sigma": sigma, "epsilon": epsilon})

        vdw_handler.slot_map.update({top_key: pot_key})
        vdw_handler.potentials.update({pot_key: pot})

        coul_handler.slot_map.update({top_key: pot_key})
        coul_handler.potentials.update(
            {pot_key: Potential(parameters={"charge": charge})}
        )

    bond_handler = SMIRNOFFBondHandler()

    for bond in structure.bonds:
        atom1 = bond.atom1
        atom2 = bond.atom2
        k = bond.type.k * kcal_mol_a2
        length = bond.type.req * unit.angstrom
        top_key = TopologyKey(atom_indices=(atom1.idx, atom2.idx))
        pot_key = PotentialKey(id=f"{atom1.idx}-{atom2.idx}")
        pot = Potential(parameters={"k": k * 2, "length": length})

        bond_handler.slot_map.update({top_key: pot_key})
        bond_handler.potentials.update({pot_key: pot})

    out.handlers.update({"vdW": vdw_handler})
    out.handlers.update({"Electrostatics": coul_handler})
    out.handlers.update({"Bonds": bond_handler})

    angle_handler = SMIRNOFFAngleHandler()

    for angle in structure.angles:
        atom1 = angle.atom1
        atom2 = angle.atom2
        atom3 = angle.atom3
        k = angle.type.k * kcal_mol_rad2
        theta = angle.type.theteq * unit.degree
        top_key = TopologyKey(atom_indices=(atom1.idx, atom2.idx, atom3.idx))
        pot_key = PotentialKey(id=f"{atom1.idx}-{atom2.idx}-{atom3.idx}")
        pot = Potential(parameters={"k": k * 2, "angle": theta})

        angle_handler.slot_map.update({top_key: pot_key})
        angle_handler.potentials.update({pot_key: pot})

    proper_torsion_handler = SMIRNOFFProperTorsionHandler()
    improper_torsion_handler = SMIRNOFFImproperTorsionHandler()

    for dihedral in structure.dihedrals:
        if isinstance(dihedral.type, pmd.DihedralType):
            if dihedral.improper:
                _process_single_dihedral(
                    dihedral, dihedral.type, improper_torsion_handler, 0
                )
            else:
                _process_single_dihedral(
                    dihedral, dihedral.type, proper_torsion_handler, 0
                )
        elif isinstance(dihedral.type, pmd.DihedralTypeList):
            for dih_idx, dihedral_type in enumerate(dihedral.type):
                if dihedral.improper:
                    _process_single_dihedral(
                        dihedral, dihedral_type, improper_torsion_handler, dih_idx
                    )
                else:
                    _process_single_dihedral(
                        dihedral,
                        dihedral_type,
                        proper_torsion_handler,
                        dih_idx,
                    )

    out.handlers.update({"Electrostatics": coul_handler})
    out.handlers.update({"Bonds": bond_handler})
    out.handlers.update({"Angles": angle_handler})
    out.handlers.update({"ProperTorsions": proper_torsion_handler})

    return out


# def _convert_box(box: unit.Quantity, structure: pmd.Structure) -> None:
def _convert_box(box, structure: "pmd.Structure") -> None:
    # TODO: Convert box vectors to box lengths + angles
    if box is None:
        lengths = [0, 0, 0]
    else:
        # TODO: Handle non-rectangular boxes
        lengths = box.diagonal().to(unit("angstrom")).magnitude
    angles = 3 * [90]
    structure.box = np.hstack([lengths, angles])


def _lj_params_from_potential(potential):
    sigma = potential.parameters["sigma"]
    epsilon = potential.parameters["epsilon"]

    return sigma, epsilon


def _process_single_dihedral(
    dihedral: "pmd.Dihedral",
    dihedral_type: "pmd.DihedralType",
    handler: Union["SMIRNOFFImproperTorsionHandler", "SMIRNOFFProperTorsionHandler"],
    mult: Optional[int] = None,
):
    atom1 = dihedral.atom1
    atom2 = dihedral.atom2
    atom3 = dihedral.atom3
    atom4 = dihedral.atom4

    k = dihedral_type.phi_k * kcal_mol_rad2
    periodicity = dihedral_type.per * unit.dimensionless
    phase = dihedral_type.phase * unit.degree
    if dihedral.improper:
        # ParmEd stores the central atom _third_ (AMBER style)
        # SMIRNOFF stores the central atom _second_
        # https://parmed.github.io/ParmEd/html/topobj/parmed.topologyobjects.Dihedral.html#parmed-topologyobjects-dihedral
        # https://open-forcefield-toolkit.readthedocs.io/en/latest/smirnoff.html#impropertorsions
        top_key = TopologyKey(
            atom_indices=(atom1.idx, atom2.idx, atom2.idx, atom4.idx),
            mult=mult,
        )
        pot_key = PotentialKey(
            id=f"{atom1.idx}-{atom3.idx}-{atom2.idx}-{atom4.idx}",
            mult=mult,
        )
        pot = Potential(parameters={"k": k, "periodicity": periodicity, "phase": phase})

        if pot_key in handler.potentials:
            raise Exception("fudging dihedral indices")

        handler.slot_map.update({top_key: pot_key})
        handler.potentials.update({pot_key: pot})
    else:
        top_key = TopologyKey(
            atom_indices=(atom1.idx, atom2.idx, atom3.idx, atom4.idx),
            mult=1,
        )
        pot_key = PotentialKey(
            id=f"{atom1.idx}-{atom2.idx}-{atom3.idx}-{atom4.idx}",
            mult=1,
        )
        pot = Potential(parameters={"k": k, "periodicity": periodicity, "phase": phase})

        while pot_key in handler.potentials:
            pot_key.mult += 1  # type: ignore[operator]
            top_key.mult += 1  # type: ignore[operator]

        handler.slot_map.update({top_key: pot_key})
        handler.potentials.update({pot_key: pot})
