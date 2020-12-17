import numpy as np
import pytest
from openforcefield.tests.utils import compare_system_energies
from openforcefield.topology import Molecule, Topology
from simtk import unit

from openff.system.stubs import ForceField


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("mol,n_mols", [("C", 1), ("CC", 1), ("C", 2), ("CC", 2)])
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
    if periodic:
        top.box_vectors = np.eye(3) * np.asarray([4, 4, 4]) * unit.nanometer
    else:
        top.box_vectors = None

    if n_mols == 1:
        positions = mol.conformers[0]
    elif n_mols == 2:
        positions = np.vstack(
            [mol.conformers[0], mol.conformers[0] + 2 * unit.nanometer]
        )

    toolkit_system = parsley.create_openmm_system(top)

    native_system = parsley.create_openff_system(top).to_openmm()

    compare_system_energies(
        system1=toolkit_system,
        system2=native_system,
        positions=positions,
        box_vectors=top.box_vectors,
    )


def test_unsupported_handler():
    """Test raising NotImplementedError when converting a system with data
    not currently supported in System.to_openmm()"""

    parsley = ForceField("openff_unconstrained-1.0.0.offxml")

    mol = Molecule.from_smiles("Cc1ccccc1")
    mol.generate_conformers(n_conformers=1)
    top = Topology.from_molecules(mol)

    with pytest.raises(NotImplementedError):
        # TODO: Catch this at openff_sys.to_openmm, not upstream
        parsley.create_openff_system(top)
