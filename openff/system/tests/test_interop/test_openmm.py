import numpy as np
import pytest
from openforcefield.topology import Molecule, Topology
from simtk import openmm, unit

from openff.system.stubs import ForceField


@pytest.mark.parametrize("mol,n_mols", [("C", 1), ("CC", 1), ("C", 2), ("CC", 2)])
def test_from_openmm_single_mols(mol, n_mols):
    """
    Test that ForceField.create_openmm_system and System.to_openmm produce
    objects with similar energies

    TODO: Tighten tolerances

    """

    parsley = ForceField("openff_unconstrained-1.0.0.offxml")

    mol = Molecule.from_smiles(mol)
    mol.generate_conformers(n_conformers=1)
    top = Topology.from_molecules(n_mols * [mol])
    top.box_vectors = np.asarray([4, 4, 4]) * unit.nanometer

    if n_mols == 1:
        positions = mol.conformers[0]
    elif n_mols == 2:
        positions = np.vstack(
            [mol.conformers[0], mol.conformers[0] + 2 * unit.nanometer]
        )

    toolkit_energy = _get_energy_from_openmm_system(
        openmm_sys=parsley.create_openmm_system(top),
        openmm_top=top.to_openmm(),
        positions=positions,
    )

    system_energy = _get_energy_from_openmm_system(
        openmm_sys=parsley.create_openff_system(top).to_openmm(),
        openmm_top=top.to_openmm(),
        positions=positions,
    )

    np.testing.assert_allclose(
        toolkit_energy / unit.kilojoule_per_mole,
        system_energy / unit.kilojoule_per_mole,
        rtol=1e-4,
        atol=1e-4,
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


def _get_energy_from_openmm_system(openmm_sys, openmm_top, positions):
    integrator = openmm.VerletIntegrator(1 * unit.femtoseconds)

    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = openmm.app.Simulation(openmm_top, openmm_sys, integrator, platform)
    simulation.context.setPositions(positions)

    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy()

    del integrator, platform, simulation, state

    return energy
