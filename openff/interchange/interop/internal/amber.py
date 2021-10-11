"""Interfaces with Amber."""
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
from openff.units import unit

from openff.interchange.components.mdtraj import _get_num_h_bonds

if TYPE_CHECKING:
    from openff.interchange.components.interchange import Interchange


AMBER_COULOMBS_CONSTANT = 18.2223
kcal_mol_a2 = unit.kilocalorie / unit.mol / unit.angstrom ** 2
kcal_mol_rad2 = unit.kilocalorie / unit.mol / unit.radian ** 2


def to_prmtop(interchange: "Interchange", file_path: Union[Path, str]):
    """
    Write a .prmtop file. See http://ambermd.org/prmtop.pdf for details.

    """
    if isinstance(file_path, str):
        path = Path(file_path)
    if isinstance(file_path, Path):
        path = file_path

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

        NATOM = interchange.topology.mdtop.n_atoms  # : total number of atoms
        NTYPES = len(typemap)  # : total number of distinct atom types
        NBONH = _get_num_h_bonds(
            interchange.topology.mdtop
        )  # : number of bonds containing hydrogen
        MBONA = (
            interchange.topology.mdtop.n_bonds - NBONH
        )  # : number of bonds not containing hydrogen
        NTHETH = 0  # : number of angles containing hydrogen
        MTHETA = 0  # : number of angles not containing hydrogen
        NPHIH = 0  # : number of dihedrals containing hydrogen
        MPHIA = 0  # : number of dihedrals not containing hydrogen
        NHPARM = 0  # : currently not used
        NPARM = 0  # : used to determine if addles created prmtop
        NNB = 0  # : number of excluded atoms
        NRES = interchange.topology.mdtop.n_residues  # : number of residues
        NBONA = 0  # : MBONA + number of constraint bonds
        NTHETA = 0  # : MTHETA + number of constraint angles
        NPHIA = 0  # : MPHIA + number of constraint dihedrals
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
        IFBOX = 0  # : set to 1 if standard periodic box, 2 when truncated octahedral
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
        for line in textwrap.wrap(text_blob, width=80, drop_whitespace=False):
            prmtop.write(line + "\n")

        prmtop.write("%FLAG ATOM_NAME\n" "%FORMAT(20a4)\n")
        text_blob = "".join([val.ljust(4) for val in typemap.values()])
        for line in textwrap.wrap(text_blob, width=80, drop_whitespace=False):
            prmtop.write(line + "\n")

        prmtop.write("%FLAG CHARGE\n" "%FORMAT(5E16.8)\n")
        charges = [
            charge.m_as(unit.e) * AMBER_COULOMBS_CONSTANT
            for charge in interchange["Electrostatics"].charges.values()
        ]
        text_blob = "".join([f"{val:16.8E}" for val in charges])
        for line in textwrap.wrap(text_blob, width=80, drop_whitespace=False):
            prmtop.write(line + "\n")

        prmtop.write("%FLAG ATOMIC_NUMBER\n" "%FORMAT(10I8)\n")
        atomic_numbers = [
            a.element.atomic_number for a in interchange.topology.mdtop.atoms
        ]
        text_blob = "".join([str(val).rjust(8) for val in atomic_numbers])
        for line in textwrap.wrap(text_blob, width=80, drop_whitespace=False):
            prmtop.write(line + "\n")

        prmtop.write("%FLAG MASS\n" "%FORMAT(5E16.8)\n")
        masses = [a.element.mass for a in interchange.topology.mdtop.atoms]
        text_blob = "".join([f"{val:16.8E}" for val in masses])
        for line in textwrap.wrap(text_blob, width=80, drop_whitespace=False):
            prmtop.write(line + "\n")

        prmtop.write("%FLAG ATOM_TYPE_INDEX\n" "%FORMAT(10I8)\n")
        unused_indices = [*range(1, NATOM + 1)]
        text_blob = "".join([str(val).rjust(8) for val in unused_indices])
        for line in textwrap.wrap(text_blob, width=80, drop_whitespace=False):
            prmtop.write(line + "\n")

        prmtop.write("%FLAG NUMBER_EXCLUDED_ATOMS\n" "%FORMAT(10I8)\n")
        number_excluded_atoms = NATOM * [0]
        text_blob = "".join([str(val).rjust(8) for val in number_excluded_atoms])
        for line in textwrap.wrap(text_blob, width=80, drop_whitespace=False):
            prmtop.write(line + "\n")

        prmtop.write("%FLAG NONBONDED_PARM_INDEX\n" "%FORMAT(10I8)\n")
        text_blob = "".join([str(val).rjust(8) for val in range(NTYPES ** 2)])
        for line in textwrap.wrap(text_blob, width=80, drop_whitespace=False):
            prmtop.write(line + "\n")

        prmtop.write("%FLAG RESIDUE_LABEL\n" "%FORMAT(20a4)\n")
        prmtop.write("FOO\n")

        prmtop.write("%FLAG REISUDE_POINTER\n" "%FORMAT(10I8)\n")
        prmtop.write("1\n")

        # TODO: Exclude (?) bonds containing hydrogens
        prmtop.write("%FLAG BOND_FORCE_CONSTANT\n" "%FORMAT(5E16.8)\n")
        bond_k = [
            interchange["Bonds"].potentials[key].parameters["k"].m_as(kcal_mol_a2) / 2
            for key in interchange["Bonds"].slot_map.values()
        ]
        text_blob = "".join([f"{val:16.8E}" for val in bond_k])
        for line in textwrap.wrap(text_blob, width=80, drop_whitespace=False):
            prmtop.write(line + "\n")

        prmtop.write("%FLAG BOND_EQUIL_VALUE\n" "%FORMAT(5E16.8)\n")
        bond_length = [
            interchange["Bonds"]
            .potentials[key]
            .parameters["length"]
            .m_as(unit.angstrom)
            for key in interchange["Bonds"].slot_map.values()
        ]
        text_blob = "".join([f"{val:16.8E}" for val in bond_length])
        for line in textwrap.wrap(text_blob, width=80, drop_whitespace=False):
            prmtop.write(line + "\n")

        prmtop.write("%FLAG ANGLE_FORCE_CONSTANT\n" "%FORMAT(5E16.8)\n")
        angle_k = [
            interchange["Angles"].potentials[key].parameters["k"].m_as(kcal_mol_rad2)
            / 2
            for key in interchange["Angles"].slot_map.values()
        ]
        text_blob = "".join([f"{val:16.8E}" for val in angle_k])
        for line in textwrap.wrap(text_blob, width=80, drop_whitespace=False):
            prmtop.write(line + "\n")

        prmtop.write("%FLAG ANGLE_EQUIL_VALUE\n" "%FORMAT(5E16.8)\n")
        angle_theta = [
            interchange["Angles"].potentials[key].parameters["angle"].m_as(unit.radian)
            for key in interchange["Angles"].slot_map.values()
        ]
        text_blob = "".join([f"{val:16.8E}" for val in angle_theta])
        for line in textwrap.wrap(text_blob, width=80, drop_whitespace=False):
            prmtop.write(line + "\n")

        prmtop.write("%FLAG DIHEDRAL_FORCE_CONSTANT\n" "%FORMAT(5E16.8)\n")
        proper_k = [
            interchange["ProperTorsions"]
            .potentials[key]
            .parameters["k"]
            .m_as(unit.kilocalorie / unit.mol)
            for key in interchange["ProperTorsions"].slot_map.values()
        ]
        text_blob = "".join([f"{val:16.8E}" for val in proper_k])
        for line in textwrap.wrap(text_blob, width=80, drop_whitespace=False):
            prmtop.write(line + "\n")

        prmtop.write("%FLAG DIHEDRAL_PERIODICITY\n" "%FORMAT(5E16.8)\n")
        proper_periodicity = [
            interchange["ProperTorsions"]
            .potentials[key]
            .parameters["periodicity"]
            .m_as(unit.dimensionless)
            for key in interchange["ProperTorsions"].slot_map.values()
        ]
        text_blob = "".join([f"{val:16.8E}" for val in proper_periodicity])
        for line in textwrap.wrap(text_blob, width=80, drop_whitespace=False):
            prmtop.write(line + "\n")

        prmtop.write("%FLAG DIHEDRAL_PHASE\n" "%FORMAT(5E16.8)\n")
        proper_phase = [
            interchange["ProperTorsions"]
            .potentials[key]
            .parameters["phase"]
            .m_as(unit.dimensionless)
            for key in interchange["ProperTorsions"].slot_map.values()
        ]
        text_blob = "".join([f"{val:16.8E}" for val in proper_phase])
        for line in textwrap.wrap(text_blob, width=80, drop_whitespace=False):
            prmtop.write(line + "\n")


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

        fmt = "%12.7f%12.7f%12.7f" "%12.7f%12.7f%12.7f\n"
        coords = interchange.positions.m_as(unit.angstrom)
        reshaped = coords.reshape((-1, 6))
        for row in reshaped:
            inpcrd.write(fmt % (row[0], row[1], row[2], row[3], row[4], row[5]))

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
