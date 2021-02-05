from typing import Dict

from simtk import unit as omm_unit


def compare_gromacs_openmm(gmx_energies: Dict, omm_energies: Dict):
    # TODO: Tighten differences
    # np.testing doesn't work on Quantity

    bond_diff = omm_energies["HarmonicBondForce"] - gmx_energies["Bond"]
    assert abs(bond_diff / omm_unit.kilojoules_per_mole) < 1e-3

    angle_diff = omm_energies["HarmonicAngleForce"] - gmx_energies["Angle"]
    assert abs(angle_diff / omm_unit.kilojoules_per_mole) < 1e-3

    torsion_diff = omm_energies["PeriodicTorsionForce"] - gmx_energies["Proper Dih."]
    assert abs(torsion_diff / omm_unit.kilojoules_per_mole) < 1e-3
