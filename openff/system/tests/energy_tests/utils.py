from typing import Dict

from simtk import unit as omm_unit


def compare_gromacs_openmm(gmx_energies: Dict, omm_energies: Dict):
    # TODO: Tighten differences
    # TODO: Nonbonded components
    # np.testing doesn't work on Quantity

    bond_diff = omm_energies["HarmonicBondForce"] - gmx_energies["Bond"]
    assert abs(bond_diff / omm_unit.kilojoules_per_mole) < 5e-3

    angle_diff = omm_energies["HarmonicAngleForce"] - gmx_energies["Angle"]
    assert abs(angle_diff / omm_unit.kilojoules_per_mole) < 5e-3

    torsion_diff = omm_energies["PeriodicTorsionForce"] - gmx_energies["Proper Dih."]
    assert abs(torsion_diff / omm_unit.kilojoules_per_mole) < 5e-3


def compare_gromacs(energies1: Dict, energies2: Dict):
    # TODO: Tighten differences
    # TODO: Nonbonded components
    # np.testing doesn't work on Quantity

    # Probably limited by ParmEd's rounding of bond parameters
    bond_diff = energies1["Bond"] - energies2["Bond"]
    assert abs(bond_diff / omm_unit.kilojoules_per_mole) < 1e-1

    angle_diff = energies1["Angle"] - energies2["Angle"]
    assert abs(angle_diff / omm_unit.kilojoules_per_mole) < 5e-3

    if "Proper Dih." in energies1.keys():
        assert "Proper Dih." in energies2.keys()
        torsion_diff = energies1["Proper Dih."] - energies2["Proper Dih."]
        assert abs(torsion_diff / omm_unit.kilojoules_per_mole) < 5e-3
