import numpy as np
import pytest
from openforcefield.tests.utils import compare_system_energies
from openforcefield.topology import Molecule, Topology
from simtk import unit

from openff.system.stubs import ForceField


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("n_mols", [1, 2])
@pytest.mark.parametrize(
    "mol",
    [
        "C",
        "CC",  # Adds a proper torsion term(s)
        "CCOC",  # This hits t86, which has a non-1.0 idivf
        "C1COC(=O)O1",  # This adds an improper, i2
    ],
)
def test_from_openmm_single_mols(periodic, mol, n_mols):
    """
    Test that ForceField.create_openmm_system and System.to_openmm produce
    objects with similar energies

    TODO: Tighten tolerances

    """

    parsley = ForceField("openff_unconstrained-1.0.0.offxml")

    mol = Molecule.from_smiles(mol)
    mol.generate_conformers(n_conformers=1)
    top = Topology.from_molecules(n_mols * [mol])
    mol.conformers[0] -= np.min(mol.conformers) * unit.angstrom

    if periodic:
        top.box_vectors = np.eye(3) * np.asarray([10, 10, 10]) * unit.nanometer
    else:
        top.box_vectors = None

    if n_mols == 1:
        positions = mol.conformers[0]
    elif n_mols == 2:
        positions = np.vstack(
            [mol.conformers[0], mol.conformers[0] + 3 * unit.nanometer]
        )
        positions = positions * unit.angstrom

    toolkit_system = parsley.create_openmm_system(top)

    native_system = parsley.create_openff_system(top).to_openmm()

    compare_system_energies(
        system1=toolkit_system,
        system2=native_system,
        positions=positions,
        box_vectors=top.box_vectors,
    )
