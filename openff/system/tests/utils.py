import importlib

import numpy as np
import pytest
from openff.toolkit.topology import Molecule, Topology
from simtk import openmm
from simtk import unit as omm_unit

from openff.system.components.system import System
from openff.system.exceptions import InterMolEnergyComparisonError


def requires_pkg(pkg_name, reason=None):
    """
    Helper function to generate a skipif decorator for any package.

    Parameters
    ----------
    pkg_name : str
        The name of the package that is required for a test(s)
    reason : str, optional
        Explanation of why the skipped it to be tested

    Returns
    -------
    requires_pkg : _pytest.mark.structures.MarkDecorator
        A pytest decorator that will skip tests if the package is not available
    """
    if not reason:
        reason = f"Package {pkg_name} is required, but was not found."
    try:
        importlib.import_module(pkg_name)
        mark = pytest.mark.skipif(
            False,
            reason="blank decorator, should never be printed",
        )
    except ImportError:
        mark = pytest.mark.skip(reason=reason)
    return mark


def top_from_smiles(
    smiles: str,
    n_molecules: int = 1,
) -> Topology:
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
    top : opennff.toolkit.topology.Topology
        A single-molecule, gas phase-like topology

    """
    mol = Molecule.from_smiles(smiles)
    mol.generate_conformers(n_conformers=1)
    top = Topology.from_molecules(n_molecules * [mol])
    # Add dummy box vectors
    # TODO: Revisit if/after Topology.is_periodic
    top.box_vectors = np.eye(3) * 10 * omm_unit.nanometer
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
        yield param[0] / omm_unit.elementary_charge


def _get_sigma_from_nonbonded_force(
    n_particles: int, nonbond_force: openmm.NonbondedForce
):
    for idx in range(n_particles):
        param = nonbond_force.getParticleParameters(idx)
        yield param[1] / omm_unit.nanometer


def _get_epsilon_from_nonbonded_force(
    n_particles: int, nonbond_force: openmm.NonbondedForce
):
    for idx in range(n_particles):
        param = nonbond_force.getParticleParameters(idx)
        yield param[2] / omm_unit.kilojoule_per_mole


def _get_lj_params_from_openmm_system(omm_sys: openmm.System):
    for force in omm_sys.getForces():
        if type(force) == openmm.NonbondedForce:
            break
    n_particles = omm_sys.getNumParticles()
    sigmas = np.asarray([*_get_sigma_from_nonbonded_force(n_particles, force)])
    epsilons = np.asarray([*_get_epsilon_from_nonbonded_force(n_particles, force)])

    return sigmas, epsilons


def _get_charges_from_openff_system(off_sys: System):
    charges_ = [*off_sys.handlers["Electrostatics"].charges.values()]  # type: ignore[attr-defined]
    charges = np.asarray([charge.magnitude for charge in charges_])
    return charges


def compare_charges_omm_off(omm_sys: openmm.System, off_sys: System) -> None:
    omm_charges = np.asarray([*_get_charges_from_openmm_system(omm_sys)])
    off_charges = _get_charges_from_openff_system(off_sys)

    np.testing.assert_equal(omm_charges, off_charges)
