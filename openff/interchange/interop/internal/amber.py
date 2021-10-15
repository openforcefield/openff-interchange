"""Interfaces with Amber."""
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Union

import numpy as np
from openff.units import unit

from openff.interchange.components.mdtraj import _get_num_h_bonds, _store_bond_partners
from openff.interchange.components.toolkit import _get_number_excluded_atoms

if TYPE_CHECKING:
    from openff.interchange.components.interchange import Interchange
    from openff.interchange.models import PotentialKey


AMBER_COULOMBS_CONSTANT = 18.2223
kcal_mol = unit.kilocalorie / unit.mol
kcal_mol_a2 = kcal_mol / unit.angstrom ** 2
kcal_mol_rad2 = kcal_mol / unit.radian ** 2


def _write_text_blob(file, blob):
    if blob == "":
        file.write("\n")
    else:
        for line in textwrap.wrap(blob, width=80, drop_whitespace=False):
            file.write(line + "\n")


def _get_exclusion_lists(topology):

    _store_bond_partners(topology.mdtop)

    number_excluded_atoms = list()
    excluded_atoms_list = list()

    for atom1 in topology.mdtop.atoms:
        # Excluded atoms _on this atom_
        tmp = list()

        for atom2 in atom1._bond_partners:

            if atom2.index > atom1.index and atom2.index + 1 not in tmp:
                tmp.append(atom2.index + 1)

            for atom3 in atom2._bond_partners:

                if atom3.index > atom1.index and atom3.index + 1 not in tmp:
                    tmp.append(atom3.index + 1)

                for atom4 in atom3._bond_partners:
                    if atom4.index > atom1.index and atom4.index + 1 not in tmp:
                        tmp.append(atom4.index + 1)

        if len(tmp) == 0:
            tmp.append(0)

        number_excluded_atoms.append(len(tmp))
        [excluded_atoms_list.append(_) for _ in tmp]

        return number_excluded_atoms, excluded_atoms_list


def to_prmtop(interchange: "Interchange", file_path: Union[Path, str]):
    """
    Write a .prmtop file. See http://ambermd.org/prmtop.pdf for details.

    """
    if isinstance(file_path, str):
        path = Path(file_path)
    if isinstance(file_path, Path):
        path = file_path

    if interchange["vdW"].mixing_rule != "lorentz-berthelot":
        raise Exception

    with open(path, "w") as prmtop:
        import datetime

        now = datetime.datetime.now()
        prmtop.write(
            "%VERSION  VERSION_STAMP = V0001.000  DATE = "
            f"{now.month:02d}/{now.day:02d}/{(now.year % 100):02}  "
            f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}\n"
            "%FLAG TITLE\n"
            "%FORMAT(20a4)\n"
            "\n"
        )

        from openff.interchange.interop.internal.gromacs import _build_typemap

        typemap = _build_typemap(interchange)  # noqa

        potential_key_to_atom_type_mapping: Dict[PotentialKey, int] = {
            key: i for i, key in enumerate(interchange["vdW"].potentials)
        }
        atom_type_indices = [
            potential_key_to_atom_type_mapping[potential_key]
            for potential_key in interchange["vdW"].slot_map.values()
        ]

        potential_key_to_bond_type_mapping: Dict[PotentialKey, int] = {
            key: i for i, key in enumerate(interchange["Bonds"].potentials)
        }

        potential_key_to_angle_type_mapping: Dict[PotentialKey, int] = {
            key: i for i, key in enumerate(interchange["Angles"].potentials)
        }

        potential_key_to_dihedral_type_mapping: Dict[PotentialKey, int] = {
            key: i for i, key in enumerate(interchange["ProperTorsions"].potentials)
        }

        bonds_inc_hydrogen = list()
        bonds_without_hydrogen = list()

        for bond, key in interchange["Bonds"].slot_map.items():
            bond_type_index = potential_key_to_bond_type_mapping[key]

            atom1 = interchange.topology.mdtop.atom(bond.atom_indices[0])
            atom2 = interchange.topology.mdtop.atom(bond.atom_indices[1])
            if atom1.element.atomic_number == 1 or atom2.element.atomic_number == 1:
                bonds_inc_hydrogen.append(atom1.index * 3)
                bonds_inc_hydrogen.append(atom2.index * 3)
                bonds_inc_hydrogen.append(bond_type_index + 1)
            else:
                bonds_without_hydrogen.append(atom1.index * 3)
                bonds_without_hydrogen.append(atom2.index * 3)
                bonds_without_hydrogen.append(bond_type_index + 1)

        angles_inc_hydrogen = list()
        angles_without_hydrogen = list()

        for angle, key in interchange["Angles"].slot_map.items():
            angle_type_index = potential_key_to_angle_type_mapping[key]

            atom1 = interchange.topology.mdtop.atom(angle.atom_indices[0])
            atom2 = interchange.topology.mdtop.atom(angle.atom_indices[1])
            atom3 = interchange.topology.mdtop.atom(angle.atom_indices[2])
            if 1 in [
                atom1.element.atomic_number,
                atom2.element.atomic_number,
                atom3.element.atomic_number,
            ]:
                angles_inc_hydrogen.append(atom1.index * 3)
                angles_inc_hydrogen.append(atom2.index * 3)
                angles_inc_hydrogen.append(atom3.index * 3)
                angles_inc_hydrogen.append(angle_type_index + 1)
            else:
                angles_without_hydrogen.append(atom1.index * 3)
                angles_without_hydrogen.append(atom2.index * 3)
                angles_without_hydrogen.append(atom3.index * 3)
                angles_without_hydrogen.append(angle_type_index + 1)

        dihedrals_inc_hydrogen = list()
        dihedrals_without_hydrogen = list()

        for dihedral, key in interchange["ProperTorsions"].slot_map.items():
            dihedral_type_index = potential_key_to_dihedral_type_mapping[key]

            atom1 = interchange.topology.mdtop.atom(dihedral.atom_indices[0])
            atom2 = interchange.topology.mdtop.atom(dihedral.atom_indices[1])
            atom3 = interchange.topology.mdtop.atom(dihedral.atom_indices[2])
            atom4 = interchange.topology.mdtop.atom(dihedral.atom_indices[2])
            if 1 in [
                atom1.element.atomic_number,
                atom2.element.atomic_number,
                atom3.element.atomic_number,
                atom4.element.atomic_number,
            ]:
                dihedrals_inc_hydrogen.append(atom1.index * 3)
                dihedrals_inc_hydrogen.append(atom2.index * 3)
                dihedrals_inc_hydrogen.append(atom3.index * 3)
                dihedrals_inc_hydrogen.append(atom4.index * 3)
                dihedrals_inc_hydrogen.append(dihedral_type_index + 1)
            else:
                dihedrals_without_hydrogen.append(atom1.index * 3)
                dihedrals_without_hydrogen.append(atom2.index * 3)
                dihedrals_without_hydrogen.append(atom3.index * 3)
                dihedrals_without_hydrogen.append(atom4.index * 3)
                dihedrals_without_hydrogen.append(dihedral_type_index + 1)

        number_excluded_atoms, excluded_atoms_list = _get_exclusion_lists(
            interchange.topology
        )
        # total number of atoms
        NATOM = interchange.topology.mdtop.n_atoms
        # total number of distinct atom types
        NTYPES = len(interchange["vdW"].potentials)
        # number of bonds containing hydrogen
        NBONH = _get_num_h_bonds(interchange.topology.mdtop)
        # number of bonds not containing hydrogen
        MBONA = interchange.topology.mdtop.n_bonds - NBONH
        # number of angles containing hydrogen
        NTHETH = int(len(angles_inc_hydrogen) / 4)
        # number of angles not containing hydrogen
        MTHETA = int(len(angles_without_hydrogen) / 4)
        # number of dihedrals containing hydrogen
        NPHIH = int(len(dihedrals_inc_hydrogen) / 5)
        # number of dihedrals not containing hydrogen
        MPHIA = int(len(dihedrals_without_hydrogen) / 5)
        NHPARM = 0  # : currently not used
        NPARM = 0  # : used to determine if addles created prmtop
        # number of excluded atoms
        NNB = len(excluded_atoms_list)
        NRES = interchange.topology.mdtop.n_residues  # : number of residues
        NBONA = MBONA  # : MBONA + number of constraint bonds
        NTHETA = MTHETA  # : MTHETA + number of constraint angles
        NPHIA = MPHIA  # : MPHIA + number of constraint dihedrals
        NUMBND = 0  # : number of unique bond types
        NUMANG = 0  # : number of unique angle types
        NPTRA = 0  # : number of unique dihedral types
        NATYP = 0  # : number of atom types in parameter file, see SOLTY below
        NPHB = 0  # : number of distinct 10-12 hydrogen bond pair types
        IFPERT = 0  # : set to 1 if perturbation info is to be read in
        NBPER = 0  # : number of bonds to be perturbed
        NGPER = 0  # : number of angles to be perturbed
        NDPER = 0  # : number of dihedrals to be perturbed
        MBPER = 0  # : number of bonds with atoms completely in perturbed group
        MGPER = 0  # : number of angles with atoms completely in perturbed group
        MDPER = 0  # : number of dihedrals with atoms completely in perturbed groups
        # set to 1 if standard periodic box, 2 when truncated octahedral
        IFBOX = 0 if interchange.box is None else 1
        NMXRS = 0  # : number of atoms in the largest residue
        IFCAP = 0  # : set to 1 if the CAP option from edit was specified
        NUMEXTRA = 0  # : number of extra points found in topology
        NCOPY = 0  # : number of PIMD slices / number of beads
        pointers = [
            NATOM,
            NTYPES,
            NBONH,
            MBONA,
            NTHETH,
            MTHETA,
            NPHIH,
            MPHIA,
            NHPARM,
            NPARM,
            NNB,
            NRES,
            NBONA,
            NTHETA,
            NPHIA,
            NUMBND,
            NUMANG,
            NPTRA,
            NATYP,
            NPHB,
            IFPERT,
            NBPER,
            NGPER,
            NDPER,
            MBPER,
            MGPER,
            MDPER,
            IFBOX,
            NMXRS,
            IFCAP,
            NUMEXTRA,
            NCOPY,
        ]

        prmtop.write("%FLAG POINTERS\n" "%FORMAT(10I8)\n")

        text_blob = "".join([str(val).rjust(8) for val in pointers])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG ATOM_NAME\n" "%FORMAT(20a4)\n")
        text_blob = "".join([val.ljust(4) for val in typemap.values()])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG CHARGE\n" "%FORMAT(5E16.8)\n")
        charges = [
            charge.m_as(unit.e) * AMBER_COULOMBS_CONSTANT
            for charge in interchange["Electrostatics"].charges.values()
        ]
        text_blob = "".join([f"{val:16.8E}" for val in charges])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG ATOMIC_NUMBER\n" "%FORMAT(10I8)\n")
        atomic_numbers = [
            a.element.atomic_number for a in interchange.topology.mdtop.atoms
        ]
        text_blob = "".join([str(val).rjust(8) for val in atomic_numbers])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG MASS\n" "%FORMAT(5E16.8)\n")
        masses = [a.element.mass for a in interchange.topology.mdtop.atoms]
        text_blob = "".join([f"{val:16.8E}" for val in masses])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG ATOM_TYPE_INDEX\n" "%FORMAT(10I8)\n")
        text_blob = "".join([str(val + 1).rjust(8) for val in atom_type_indices])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG NUMBER_EXCLUDED_ATOMS\n" "%FORMAT(10I8)\n")
        # This approach assumes ordering, 0 index, etc.
        number_excluded_atoms = _get_number_excluded_atoms(
            interchange.topology
        ).values()

        # https://ambermd.org/prmtop.pdf says this section is ignored (!?)
        number_excluded_atoms = NATOM * [0]
        text_blob = "".join([str(val).rjust(8) for val in number_excluded_atoms])
        _write_text_blob(prmtop, text_blob)

        acoefs = list()
        bcoefs = list()
        # index = NONBONDED PARM INDEX [NTYPES × (ATOM TYPE INDEX(i) − 1) + ATOM TYPE INDEX(j)]
        nonbonded_parm_atom_type_tuple_mappings = dict()
        for i, key_i in enumerate(potential_key_to_atom_type_mapping):
            for j, key_j in enumerate(potential_key_to_atom_type_mapping):

                index = NTYPES * (i) + j + 1
                # TODO: Figure out the right way to map cross-interactions, using the
                #       key_i and key_j objects as lookups to parameters
                sigma_i = interchange["vdW"].potentials[key_i].parameters["sigma"]
                sigma_j = interchange["vdW"].potentials[key_j].parameters["sigma"]
                epsilon_i = interchange["vdW"].potentials[key_i].parameters["epsilon"]
                epsilon_j = interchange["vdW"].potentials[key_j].parameters["epsilon"]

                sigma = (sigma_i + sigma_j) * 0.5
                epsilon = (epsilon_i * epsilon_j) ** 0.5

                acoef = (4 * epsilon * sigma ** 12).m_as(kcal_mol * unit.angstrom ** 12)
                bcoef = (4 * epsilon * sigma ** 6).m_as(kcal_mol * unit.angstrom ** 6)

                # TODO: This is probably dangerous on the basis that it's likely
                #       sensitive to rounding
                if acoef not in acoefs:
                    acoefs.append(acoef)
                    bcoefs.append(bcoef)

                index = acoefs.index(acoef)

                nonbonded_parm_atom_type_tuple_mappings[tuple((key_i, key_j))] = (
                    index + 1
                )

        prmtop.write("%FLAG NONBONDED_PARM_INDEX\n" "%FORMAT(10I8)\n")
        text_blob = "".join(
            [
                str(val).rjust(8)
                for val in nonbonded_parm_atom_type_tuple_mappings.values()
            ]
        )
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG RESIDUE_LABEL\n" "%FORMAT(20a4)\n")
        prmtop.write("FOO\n")

        prmtop.write("%FLAG RESIDUE_POINTER\n" "%FORMAT(10I8)\n")
        prmtop.write("       1\n")

        # TODO: Exclude (?) bonds containing hydrogens
        prmtop.write("%FLAG BOND_FORCE_CONSTANT\n" "%FORMAT(5E16.8)\n")
        bond_k = [
            interchange["Bonds"].potentials[key].parameters["k"].m_as(kcal_mol_a2) / 2
            for key in potential_key_to_bond_type_mapping
        ]
        text_blob = "".join([f"{val:16.8E}" for val in bond_k])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG BOND_EQUIL_VALUE\n" "%FORMAT(5E16.8)\n")
        bond_length = [
            interchange["Bonds"]
            .potentials[key]
            .parameters["length"]
            .m_as(unit.angstrom)
            for key in potential_key_to_bond_type_mapping
        ]
        text_blob = "".join([f"{val:16.8E}" for val in bond_length])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG ANGLE_FORCE_CONSTANT\n" "%FORMAT(5E16.8)\n")
        angle_k = [
            interchange["Angles"].potentials[key].parameters["k"].m_as(kcal_mol_rad2)
            / 2  # noqa
            for key in interchange["Angles"].slot_map.values()
        ]
        text_blob = "".join([f"{val:16.8E}" for val in angle_k])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG ANGLE_EQUIL_VALUE\n" "%FORMAT(5E16.8)\n")
        angle_theta = [
            interchange["Angles"].potentials[key].parameters["angle"].m_as(unit.radian)
            for key in interchange["Angles"].slot_map.values()
        ]
        text_blob = "".join([f"{val:16.8E}" for val in angle_theta])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG DIHEDRAL_FORCE_CONSTANT\n" "%FORMAT(5E16.8)\n")
        proper_k = [
            interchange["ProperTorsions"]
            .potentials[key]
            .parameters["k"]
            .m_as(unit.kilocalorie / unit.mol)
            for key in interchange["ProperTorsions"].slot_map.values()
        ]
        text_blob = "".join([f"{val:16.8E}" for val in proper_k])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG DIHEDRAL_PERIODICITY\n" "%FORMAT(5E16.8)\n")
        proper_periodicity = [
            interchange["ProperTorsions"]
            .potentials[key]
            .parameters["periodicity"]
            .m_as(unit.dimensionless)
            for key in interchange["ProperTorsions"].slot_map.values()
        ]
        text_blob = "".join([f"{val:16.8E}" for val in proper_periodicity])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG DIHEDRAL_PHASE\n" "%FORMAT(5E16.8)\n")
        proper_phase = [
            interchange["ProperTorsions"]
            .potentials[key]
            .parameters["phase"]
            .m_as(unit.dimensionless)
            for key in interchange["ProperTorsions"].slot_map.values()
        ]
        text_blob = "".join([f"{val:16.8E}" for val in proper_phase])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG SCEE_SCALE_FACTOR\n" "%FORMAT(5E16.8)\n")
        scee = len(interchange["ProperTorsions"].slot_map) * [1.2]
        text_blob = "".join([f"{val:16.8E}" for val in scee])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG SCNB_SCALE_FACTOR\n" "%FORMAT(5E16.8)\n")
        scnb = len(interchange["ProperTorsions"].slot_map) * [2.0]
        text_blob = "".join([f"{val:16.8E}" for val in scnb])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG SOLTY\n" "%FORMAT(5E16.8)\n")
        prmtop.write(f"{0:16.8E}\n")

        prmtop.write("%FLAG LENNARD_JONES_ACOEF\n" "%FORMAT(5E16.8)\n")
        text_blob = "".join([f"{val:16.8E}" for val in acoefs])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG LENNARD_JONES_BCOEF\n" "%FORMAT(5E16.8)\n")
        text_blob = "".join([f"{val:16.8E}" for val in bcoefs])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG BONDS_INC_HYDROGEN\n" "%FORMAT(10I8)\n")
        text_blob = "".join([str(val).rjust(8) for val in bonds_inc_hydrogen])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG BONDS_WITHOUT_HYDROGEN\n" "%FORMAT(10I8)\n")
        text_blob = "".join([str(val).rjust(8) for val in bonds_without_hydrogen])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG ANGLES_INC_HYDROGEN\n" "%FORMAT(10I8)\n")
        text_blob = "".join([str(val).rjust(8) for val in angles_inc_hydrogen])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG ANGLES_WITHOUT_HYDROGEN\n" "%FORMAT(10I8)\n")
        text_blob = "".join([str(val).rjust(8) for val in angles_without_hydrogen])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG DIHEDRALS_INC_HYDROGEN\n" "%FORMAT(10I8)\n")
        text_blob = "".join([str(val).rjust(8) for val in dihedrals_inc_hydrogen])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG DIHEDRALS_WITHOUT_HYDROGEN\n" "%FORMAT(10I8)\n")
        text_blob = "".join([str(val).rjust(8) for val in dihedrals_without_hydrogen])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG EXCLUDED_ATOMS_LIST\n" "%FORMAT(10I8)\n")
        _write_text_blob(prmtop, "")

        prmtop.write("%FLAG HBOND_ACOEF\n" "%FORMAT(5E16.8)\n")
        _write_text_blob(prmtop, "")

        prmtop.write("%FLAG HBOND_BCOEF\n" "%FORMAT(5E16.8)\n")
        _write_text_blob(prmtop, "")

        prmtop.write("%FLAG HBCUT\n" "%FORMAT(5E16.8)\n")
        _write_text_blob(prmtop, "")

        prmtop.write("%FLAG AMBER_ATOM_TYPE\n" "%FORMAT(20a4)\n")
        text_blob = "".join([val.ljust(4) for val in typemap.values()])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG TREE_CHAIN_CLASSIFICATION\n" "%FORMAT(20a4)\n")
        blahs = NATOM * ["BLA"]
        text_blob = "".join([val.ljust(4) for val in blahs])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG JOIN_ARRAY\n" "%FORMAT(10I8)\n")
        _ = NATOM * [0]
        text_blob = "".join([str(val).rjust(8) for val in _])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG IROTAT\n" "%FORMAT(10I8)\n")
        _ = NATOM * [0]
        text_blob = "".join([str(val).rjust(8) for val in _])
        _write_text_blob(prmtop, text_blob)

        if IFBOX == 1:
            prmtop.write("%FLAG SOLVENT_POINTERS\n" "%FORMAT(3I8)\n")
            prmtop.write("       1       1       2\n")

            # TODO: No easy way to accurately export this section while
            #       using an MDTraj topology
            prmtop.write("%FLAG ATOMS_PER_MOLECULE\n" "%FORMAT(10I8)\n")
            prmtop.write(str(interchange.topology.mdtop.n_atoms).rjust(8))
            prmtop.write("\n")

            prmtop.write("%FLAG BOX_DIMENSIONS\n" "%FORMAT(5E16.8)\n")
            box = [90.0]
            for i in range(3):
                box.append(interchange.box[i, i].m_as(unit.angstrom))
            text_blob = "".join([f"{val:16.8E}" for val in box])
            _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG RADIUS_SET\n" "%FORMAT(1a80)\n")
        prmtop.write("0\n")

        prmtop.write("%FLAG RADII\n" "%FORMAT(5E16.8)\n")
        radii = NATOM * [0]
        text_blob = "".join([f"{val:16.8E}" for val in radii])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG SCREEN\n" "%FORMAT(5E16.8)\n")
        screen = NATOM * [0]
        text_blob = "".join([f"{val:16.8E}" for val in screen])
        _write_text_blob(prmtop, text_blob)

        prmtop.write("%FLAG IPOL\n" "%FORMAT(1I8)\n")
        prmtop.write("       0\n")


def to_inpcrd(interchange: "Interchange", file_path: Union[Path, str]):
    """
    Write a .prmtop file. See https://ambermd.org/FileFormats.php#restart for details.

    """
    if isinstance(file_path, str):
        path = Path(file_path)
    if isinstance(file_path, Path):
        path = file_path

    n_atoms = interchange.topology.mdtop.n_atoms  # type: ignore
    time = 0.0

    with open(path, "w") as inpcrd:
        inpcrd.write(f"\n{n_atoms:5d}{time:15.7e}\n")

        coords = interchange.positions.m_as(unit.angstrom)
        blob = "".join([f"{val:12.7f}".rjust(12) for val in coords.flatten()])

        for line in textwrap.wrap(blob, width=72, drop_whitespace=False):
            inpcrd.write(line + "\n")

        # fmt = "%12.7f%12.7f%12.7f" "%12.7f%12.7f%12.7f\n"
        # reshaped = coords.reshape((-1, 6))
        # for row in reshaped:
        #     inpcrd.write(fmt % (row[0], row[1], row[2], row[3], row[4], row[5]))

        box = interchange.box.to(unit.angstrom).magnitude
        if (box == np.diag(np.diagonal(box))).all():
            for i in range(3):
                inpcrd.write(f"{box[i, i]:12.7f}")
            for _ in range(3):
                inpcrd.write("  90.0000000")
        else:
            # TODO: Handle non-rectangular
            raise NotImplementedError

        inpcrd.write("\n")
