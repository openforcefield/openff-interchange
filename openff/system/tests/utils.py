from collections import defaultdict
from typing import Dict, List, Tuple

import mdtraj as md
import numpy as np
import pytest
from openff.toolkit.topology import Molecule
from openff.utilities.utilities import has_executable
from simtk import openmm
from simtk import unit as simtk_unit

from openff.system.components.mdtraj import OFFBioTop
from openff.system.components.system import System
from openff.system.exceptions import InterMolEnergyComparisonError

HAS_GROMACS = any(has_executable(e) for e in ["gmx", "gmx_d"])
HAS_LAMMPS = any(has_executable(e) for e in ["lammps", "lmp_mpi", "lmp_serial"])

needs_gmx = pytest.mark.skipif(not HAS_GROMACS, reason="Needs GROMACS")
needs_lmp = pytest.mark.skipif(not HAS_LAMMPS, reason="Needs GROMACS")


kj_nm2_mol = simtk_unit.kilojoule_per_mole / simtk_unit.nanometer ** 2
kj_rad2_mol = simtk_unit.kilojoule_per_mole / simtk_unit.radian ** 2


def top_from_smiles(
    smiles: str,
    n_molecules: int = 1,
) -> OFFBioTop:
    """Create a gas phase OpenFF Topology from a single-molecule SMILES

    Parameters
    ----------
    smiles : str
        The SMILES of the input molecule
    n_molecules : int, optional, default = 1
        The number of copies of the SMILES molecule from which to
        compose a topology

    Returns
    -------
    top : opennff.system.components.mdtraj.OFFBioTop
        A single-molecule, gas phase-like topology

    """
    mol = Molecule.from_smiles(smiles)
    mol.generate_conformers(n_conformers=1)
    top = OFFBioTop.from_molecules(n_molecules * [mol])
    top.mdtop = md.Topology.from_openmm(top.to_openmm())  # type: ignore[attr-defined]
    # Add dummy box vectors
    # TODO: Revisit if/after Topology.is_periodic
    top.box_vectors = np.eye(3) * 10 * simtk_unit.nanometer
    return top


def compare_energies(ener1, ener2, atol=1e-8):
    """Compare two GROMACS energy dicts from InterMol"""

    assert sorted(ener1.keys()) == sorted(ener2.keys()), (
        sorted(ener1.keys()),
        sorted(ener2.keys()),
    )

    flaky_keys = [
        "Temperature",
        "Kinetic En.",
        "Total Energy",
        "Pressure",
        "Vir-XX",
        "Vir-YY",
    ]

    failed_runs = []
    for key in ener1.keys():
        if key in flaky_keys:
            continue
        try:
            assert np.isclose(
                ener1[key] / ener1[key].unit,
                ener2[key] / ener2[key].unit,
                atol=atol,
            )
        except AssertionError:
            failed_runs.append([key, ener1[key], ener2[key]])

    if len(failed_runs) > 0:
        raise InterMolEnergyComparisonError(failed_runs)


def _get_charges_from_openmm_system(omm_sys: openmm.System):
    for force in omm_sys.getForces():
        if type(force) == openmm.NonbondedForce:
            break
    for idx in range(omm_sys.getNumParticles()):
        param = force.getParticleParameters(idx)
        yield param[0].value_in_unit(simtk_unit.elementary_charge)


def _get_sigma_from_nonbonded_force(
    n_particles: int, nonbond_force: openmm.NonbondedForce
):
    for idx in range(n_particles):
        param = nonbond_force.getParticleParameters(idx)
        yield param[1].value_in_unit(simtk_unit.nanometer)


def _get_epsilon_from_nonbonded_force(
    n_particles: int, nonbond_force: openmm.NonbondedForce
):
    for idx in range(n_particles):
        param = nonbond_force.getParticleParameters(idx)
        yield param[2].value_in_unit(simtk_unit.kilojoule_per_mole)


def _get_lj_params_from_openmm_system(omm_sys: openmm.System):
    for force in omm_sys.getForces():
        if type(force) == openmm.NonbondedForce:
            break
    n_particles = omm_sys.getNumParticles()
    sigmas = np.asarray([*_get_sigma_from_nonbonded_force(n_particles, force)])
    epsilons = np.asarray([*_get_epsilon_from_nonbonded_force(n_particles, force)])

    return sigmas, epsilons


def _get_charges_from_openff_system(off_sys: System):
    charges_ = [*off_sys.handlers["Electrostatics"].charges.values()]
    charges = np.asarray([charge.magnitude for charge in charges_])
    return charges


def compare_charges_omm_off(omm_sys: openmm.System, off_sys: System) -> None:
    omm_charges = np.asarray([*_get_charges_from_openmm_system(omm_sys)])
    off_charges = _get_charges_from_openff_system(off_sys)

    np.testing.assert_equal(omm_charges, off_charges)


def _create_torsion_dict(torsion_force) -> Dict[Tuple[int], List[Tuple]]:
    torsions = defaultdict(list)

    for i in range(torsion_force.getNumTorsions()):
        p1, p2, p3, p4, periodicity, phase, k = torsion_force.getTorsionParameters(i)
        key = (p1, p2, p3, p4)
        torsions[key]
        torsions[key].append((periodicity, phase, k))

    return torsions


def _create_bond_dict(bond_force):
    bonds = dict()

    for i in range(bond_force.getNumBonds()):
        p1, p2, length, k = bond_force.getBondParameters(i)
        key = (p1, p2)
        bonds[key] = (length, k)

    return bonds


def _create_angle_dict(angle_force):
    angles = dict()

    for i in range(angle_force.getNumAngles()):
        p1, p2, p3, theta, k = angle_force.getAngleParameters(i)
        key = (p1, p2, p3)
        angles[key] = (theta, k)

    return angles


def _compare_individual_torsions(x, y):
    assert x[0] == y[0]
    assert x[1] == y[1]
    assert (x[2] - y[2]) < 1e-15 * simtk_unit.kilojoule_per_mole


def _compare_torsion_forces(force1, force2):
    sorted1 = _create_torsion_dict(torsion_force=force1)
    sorted2 = _create_torsion_dict(torsion_force=force2)

    assert sum(len(v) for v in sorted1.values()) == force1.getNumTorsions()
    assert sum(len(v) for v in sorted2.values()) == force2.getNumTorsions()
    assert len(sorted1) == len(sorted2)

    for key in sorted1:
        for i in range(len(sorted1[key])):
            _compare_individual_torsions(sorted1[key][i], sorted2[key][i])


def _compare_bond_forces(force1, force2):
    assert force1.getNumBonds() == force2.getNumBonds()

    bonds1 = _create_bond_dict(force1)
    bonds2 = _create_bond_dict(force2)

    for key in bonds1:
        assert abs(bonds2[key][0] - bonds1[key][0]) < 1e-15 * simtk_unit.nanometer
        assert abs(bonds2[key][1] - bonds1[key][1]) < 1e-9 * kj_nm2_mol, abs(
            bonds2[key][1] - bonds1[key][1]
        )


def _compare_angle_forces(force1, force2):
    assert force1.getNumAngles() == force2.getNumAngles()

    angles1 = _create_angle_dict(force1)
    angles2 = _create_angle_dict(force2)

    for key in angles1:
        assert abs(angles2[key][0] - angles1[key][0]) < 1e-15 * simtk_unit.radian
        assert abs(angles2[key][1] - angles1[key][1]) < 1e-10 * kj_rad2_mol


def _compare_nonbonded_settings(force1, force2):
    for attr in dir(force1):
        if not attr.startswith("get") or attr in [
            "getExceptionParameterOffset",
            "getExceptionParameters",
            "getGlobalParameterDefaultValue",
            "getGlobalParameterName",
            "getLJPMEParametersInContext",
            "getPMEParametersInContext",
            "getParticleParameterOffset",
            "getParticleParameters",
        ]:
            continue
        assert getattr(force1, attr)() == getattr(force2, attr)(), attr


def _compare_nonbonded_parameters(force1, force2):
    assert force1.getNumParticles() == force2.getNumParticles()

    for i in range(force1.getNumParticles()):
        q1, sig1, eps1 = force1.getParticleParameters(i)
        q2, sig2, eps2 = force2.getParticleParameters(i)
        assert abs(q2 - q1) < 1e-12 * simtk_unit.elementary_charge
        assert abs(sig2 - sig1) < 1e-12 * simtk_unit.nanometer
        assert abs(eps2 - eps1) < 1e-12 * simtk_unit.kilojoule_per_mole


def _compare_exceptions(force1, force2):
    assert force1.getNumExceptions() == force2.getNumExceptions()

    for i in range(force1.getNumExceptions()):
        _, _, q1, sig1, eps1 = force1.getExceptionParameters(i)
        _, _, q2, sig2, eps2 = force2.getExceptionParameters(i)
        assert abs(q2 - q1) < 1e-12 * simtk_unit.elementary_charge ** 2
        assert abs(sig2 - sig1) < 1e-12 * simtk_unit.nanometer
        assert abs(eps2 - eps1) < 1e-12 * simtk_unit.kilojoule_per_mole


def _get_force(openmm_sys: openmm.System, force_type):
    forces = [f for f in openmm_sys.getForces() if type(f) == force_type]

    if len(forces) > 1:
        raise NotImplementedError("Not yet able to process duplicate forces types")
    return forces[0]
