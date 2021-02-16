import importlib

import numpy as np
import pytest
from openff.toolkit.topology import Molecule, Topology
from simtk import unit

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
    top.box_vectors = np.eye(3) * 10 * unit.nanometer
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
