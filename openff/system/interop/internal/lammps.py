from pathlib import Path
from typing import IO, Dict, Union

import numpy as np
from simtk import unit as omm_unit

from openff.system import unit
from openff.system.components.system import System


def to_lammps(openff_sys: System, file_path: Union[Path, str]):

    if isinstance(file_path, str):
        path = Path(file_path)
    if isinstance(file_path, Path):
        path = file_path

    n_atoms = openff_sys.topology.n_topology_atoms  # type: ignore[union-attr]
    if "Bonds" in openff_sys.handlers:
        n_bonds = len(openff_sys["Bonds"].slot_map.keys())
    else:
        n_bonds = 0
    if "Angles" in openff_sys.handlers:
        n_angles = len(openff_sys["Angles"].slot_map.keys())
    else:
        n_angles = 0
    if "ProperTorsions" in openff_sys.handlers:
        n_propers = len(openff_sys["ProperTorsions"].slot_map.keys())
    else:
        n_propers = 0
    if "ImproperTorsions" in openff_sys.handlers:
        n_impropers = len(openff_sys["ImproperTorsions"].slot_map.keys())
    else:
        n_impropers = 0

    with open(path, "w") as lmp_file:
        lmp_file.write("Title\n\n")

        lmp_file.write(f"{n_atoms} atoms\n")
        lmp_file.write(f"{n_bonds} bonds\n")
        lmp_file.write(f"{n_angles} angles\n")
        lmp_file.write(f"{n_propers} dihedrals\n")
        lmp_file.write(f"{n_impropers} impropers\n")

        lmp_file.write("\n1 atom types\n")

        lmp_file.write("\n")

        # write types section

        x_min, y_min, z_min = np.min(
            openff_sys.positions.to(unit.angstrom), axis=0  # type: ignore[attr-defined]
        ).magnitude
        L_x, L_y, L_z = np.diag(openff_sys.box.to(unit.angstrom).magnitude)  # type: ignore[attr-defined]

        lmp_file.write(
            "{:.10g} {:.10g} xlo xhi\n"
            "{:.10g} {:.10g} ylo yhi\n"
            "{:.10g} {:.10g} zlo zhi\n".format(
                x_min,
                x_min + L_x,
                y_min,
                y_min + L_y,
                z_min,
                z_min + L_z,
            )
        )

        lmp_file.write("0.0 0.0 0.0 xy xz yz\n")

        lmp_file.write("\nMasses\n\n")

        for atom_idx, atom in enumerate(openff_sys.topology.topology_atoms):  # type: ignore[union-attr]
            mass = atom.atom.mass / omm_unit.dalton
            lmp_file.write("{:d}\t{:.8g}".format(atom_idx + 1, mass))

        lmp_file.write("\n\n")

        atom_type_map = _write_pair_coeffs(lmp_file=lmp_file, openff_sys=openff_sys)

        if n_bonds > 0:
            _write_bond_coeffs(lmp_file=lmp_file, openff_sys=openff_sys)
        if n_angles > 0:
            _write_angle_coeffs(lmp_file=lmp_file, openff_sys=openff_sys)
        if n_propers > 0 or n_impropers > 0:
            pass

        _write_atoms(
            lmp_file=lmp_file, openff_sys=openff_sys, atom_type_map=atom_type_map
        )


def _write_pair_coeffs(lmp_file: IO, openff_sys: System) -> Dict:
    lmp_file.write("Pair Coeffs\n\n")

    vdw_handler = openff_sys["vdW"]
    atom_type_map = dict(enumerate(vdw_handler.potentials))

    for atom_type_idx, smirks in atom_type_map.items():
        params = vdw_handler.potentials[smirks].parameters

        sigma = params["sigma"].to(unit.angstrom).magnitude
        epsilon = params["epsilon"].to(unit.Unit("kilocalorie / mole")).magnitude

        lmp_file.write(
            "{:d}\t{:.8g}\t{:.8g}\n".format(atom_type_idx + 1, epsilon, sigma)
        )

    lmp_file.write("\n")

    return atom_type_map


def _write_bond_coeffs(lmp_file: IO, openff_sys: System) -> Dict:
    lmp_file.write("Bond Coeffs\n\n")

    bond_handler = openff_sys.handlers["Bonds"]
    bond_type_map = dict(enumerate(bond_handler.potentials))

    for bond_type_idx, smirks in bond_type_map.items():
        params = bond_handler.potentials[smirks].parameters

        k = params["k"].to(unit.Unit("kilocalorie / mole / angstrom ** 2")).magnitude
        k = k * 0.5  # Account for LAMMPS wrapping 1/2 into k
        length = params["length"].to(unit.angstrom).magnitude

        lmp_file.write(f"{bond_type_idx:d} harmonic\t{k:.16g}\t{length:.16g}")

    lmp_file.write("\n")

    return bond_type_map


def _write_angle_coeffs(lmp_file: IO, openff_sys: System) -> Dict:
    lmp_file.write("Angle Coeffs\n\n")

    angle_handler = openff_sys.handlers["Angles"]
    angle_type_map = dict(enumerate(angle_handler.potentials))

    for angle_type_idx, smirks in angle_type_map.items():
        params = angle_handler.potentials[smirks].parameters

        k = params["k"].to(unit.Unit("kilocalorie / mole / angstrom ** 2")).magnitude
        k = k * 0.5  # Account for LAMMPS wrapping 1/2 into k
        theta = params["angle"].to(unit.degree).magnitude

        lmp_file.write(f"{angle_type_idx:d} harmonic\t{k:.16g}\t{theta:.16g}")

    lmp_file.write("\n")

    return angle_type_map


def _write_atoms(lmp_file: IO, openff_sys: System, atom_type_map: Dict):
    lmp_file.write("Atoms\n\n")

    molecule_map = dict(enumerate(openff_sys.topology.topology_molecules))  # type: ignore[union-attr]
    molecule_map_inv = dict({v: k for k, v in molecule_map.items()})

    atom_type_map_inv = dict({v: k for k, v in atom_type_map.items()})

    electrostatics_handler = openff_sys.handlers["Electrostatics"]
    vdw_hander = openff_sys.handlers["vdW"]

    for atom_idx, atom in enumerate(openff_sys.topology.topology_atoms):  # type: ignore[union-attr]

        molecule_idx = molecule_map_inv[atom.topology_molecule]

        vdw_smirks = vdw_hander.slot_map[str((atom_idx,))]
        atom_type = atom_type_map_inv[vdw_smirks]

        charge = electrostatics_handler.charges[str((atom_idx,))].magnitude  # type: ignore
        pos = openff_sys.positions[atom_idx].to(unit.angstrom).magnitude
        lmp_file.write(
            "{:d}\t{:d}\t{:d}\t{:.8g}\t{:.8g}\t{:.8g}\t{:.8g}\n".format(
                atom_idx + 1,
                molecule_idx + 1,
                atom_type + 1,
                charge,
                pos[0],
                pos[1],
                pos[2],
            )
        )
