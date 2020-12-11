import numpy as np
from openforcefield.topology import Molecule, Topology
from simtk import openmm, unit

from openff.system.stubs import ForceField


def test_from_openmm():
    parsley = ForceField("openff_unconstrained-1.0.0.offxml")

    mol = Molecule.from_smiles("C")
    mol.generate_conformers(n_conformers=1)
    top = Topology.from_molecules([mol])
    top.box_vectors = np.asarray([4, 4, 4]) * unit.nanometer

    toolkit_energy = _get_energy_from_openmm_system(
        openmm_sys=parsley.create_openmm_system(top),
        openmm_top=top.to_openmm(),
        positions=mol.conformer[0],
    )

    system_energy = _get_energy_from_openmm_system(
        openmm_sys=parsley.create_openff_system(top).to_openmm(),
        openmm_top=top.to_openmm(),
        positions=mol.conformer[0],
    )

    assert toolkit_energy == system_energy


def _get_energy_from_openmm_system(openmm_sys, openmm_top, positions):
    integrator = openmm.VerletIntegrator(1 * unit.femtoseconds)

    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = openmm.app.Simulation(openmm_top, openmm_sys, integrator, platform)
    simulation.context.setPositions(positions)

    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy()

    del integrator, platform, simulation, state

    return energy
