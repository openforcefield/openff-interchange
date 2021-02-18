from typing import Dict

from simtk import unit as omm_unit

from openff.system.exceptions import NonbondedEnergyError


def compare_gromacs_openmm(
    gmx_energies: Dict,
    omm_energies: Dict,
    custom_tolerances: Dict[str, float] = None,
):
    # TODO: Tighten differences
    # TODO: Nonbonded components
    # np.testing doesn't work on Quantity

    tolerances = {
        "Bond": 1e-3,
        "Angle": 1e-3,
        "Torsion": 1e-3,
        "Nonbonded": 1e-3,
    }

    tolerances.update(custom_tolerances)

    if "Bond" in gmx_energies.keys():
        bond_diff = omm_energies["HarmonicBondForce"] - gmx_energies["Bond"]
        bond_diff /= omm_unit.kilojoules_per_mole
        assert abs(bond_diff) < tolerances["Bond"], bond_diff

    if "Angle" in gmx_energies.keys():
        angle_diff = omm_energies["HarmonicAngleForce"] - gmx_energies["Angle"]
        angle_diff /= omm_unit.kilojoules_per_mole
        assert abs(angle_diff) < tolerances["Angle"], angle_diff

    if "Proper Dih." in gmx_energies.keys():
        torsion_diff = (
            omm_energies["PeriodicTorsionForce"] - gmx_energies["Proper Dih."]
        )
        torsion_diff /= omm_unit.kilojoules_per_mole
        assert abs(torsion_diff) < tolerances["Torsion"], torsion_diff

    gmx_nonbonded = _get_gmx_energy_nonbonded(gmx_energies)
    nonbonded_diff = omm_energies["NonbondedForce"] - gmx_nonbonded
    if abs(nonbonded_diff / omm_unit.kilojoules_per_mole) > tolerances["Nonbonded"]:
        raise NonbondedEnergyError(nonbonded_diff)


def compare_gromacs(
    energies1: Dict,
    energies2: Dict,
    custom_tolerances: Dict[str, float] = None,
):
    # TODO: Tighten differences
    # TODO: Nonbonded components
    # np.testing doesn't work on Quantity

    tolerances = {
        "Bond": 1e-1,
        "Angle": 5 - 3,
        "Torsion": 5e-3,
        "Nonbonded": 1e-3,
    }

    tolerances.update(custom_tolerances)

    # Probably limited by ParmEd's rounding of bond parameters
    bond_diff = energies1["Bond"] - energies2["Bond"]
    assert abs(bond_diff / omm_unit.kilojoules_per_mole) < 1e-1

    angle_diff = energies1["Angle"] - energies2["Angle"]
    assert abs(angle_diff / omm_unit.kilojoules_per_mole) < 5e-3

    if "Proper Dih." in energies1.keys():
        assert "Proper Dih." in energies2.keys()
        torsion_diff = energies1["Proper Dih."] - energies2["Proper Dih."]
        assert abs(torsion_diff / omm_unit.kilojoules_per_mole) < 5e-3

    # TODO: Fix constraints and other issues around GROMACS non-bonded energies


def compare_openmm(energies1: Dict, energies2: Dict):
    for key, val in energies1.items():
        if energies1[key]._value == 0.0:
            continue
        energy_diff = val - energies2[key]
        if abs(energy_diff / val.unit) > 5e-3:
            if key == "NonbondedForce":
                raise NonbondedEnergyError(energy_diff)
            else:
                raise AssertionError(key, energy_diff)


def _get_gmx_energy_nonbonded(gmx_energies: Dict):
    """Get the total nonbonded energy from a set of GROMACS energies"""
    gmx_nonbonded = 0 * gmx_energies["Potential"].unit
    for key in ["LJ (SR)", "Coulomb (SR)", "Coul. recip.", "Disper. corr."]:
        try:
            gmx_nonbonded += gmx_energies[key]
        except KeyError:
            pass

    return gmx_nonbonded
