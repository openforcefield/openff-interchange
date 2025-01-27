import random

import pytest
from openff.toolkit import Molecule, Quantity, Topology

from openff.interchange._tests import MoleculeWithConformer
from openff.interchange.exceptions import NegativeMassError


@pytest.mark.parametrize("reversed", [False, True])
def test_hmr_basic(sage, reversed, ethanol, reversed_ethanol):
    pytest.importorskip("openmm.unit")
    import openmm.unit

    hydrogen_mass = random.uniform(1.0, 4.0)

    molecule = reversed_ethanol if reversed else ethanol
    molecule.generate_conformers(n_conformers=1)

    topology = molecule.to_topology()

    interchange = sage.create_interchange(topology)

    system = interchange.to_openmm(hydrogen_mass=hydrogen_mass)

    expected_mass = sum([atom.mass for atom in topology.atoms]).m_as("dalton")

    found_mass = sum(
        [
            system.getParticleMass(particle_index).value_in_unit(openmm.unit.dalton)
            for particle_index in range(system.getNumParticles())
        ],
    )

    assert found_mass == pytest.approx(expected_mass)

    for particle_index, atom in enumerate(topology.atoms):
        if atom.atomic_number == 1:
            assert system.getParticleMass(particle_index)._value == hydrogen_mass


def test_hmr_not_applied_to_water(sage, water):
    # TODO: This should have different behavior for rigid and flexible water,
    #       but sage has tip3p (rigid) so it should always be skipped

    pytest.importorskip("openmm.unit")
    import openmm.unit
    from openmm.app import element

    hydrogen_mass = 1.23

    interchange = sage.create_interchange(water.to_topology())

    system = interchange.to_openmm(hydrogen_mass=hydrogen_mass)

    expected_mass = sum([atom.mass for atom in interchange.topology.atoms]).m_as("dalton")

    found_mass = sum(
        [
            system.getParticleMass(particle_index).value_in_unit(openmm.unit.dalton)
            for particle_index in range(system.getNumParticles())
        ],
    )

    assert found_mass == pytest.approx(expected_mass)

    for particle_index, atom in enumerate(interchange.topology.atoms):
        if atom.atomic_number == 1:
            assert system.getParticleMass(particle_index) == element.hydrogen.mass


def test_mass_is_positive(sage):
    pytest.importorskip("openmm")

    with pytest.raises(
        NegativeMassError,
        match="Particle with index 0 would have a negative mass after.*5.0",
    ):
        sage.create_interchange(Molecule.from_smiles("C").to_topology()).to_openmm(hydrogen_mass=5.0)


@pytest.mark.parametrize("hydrogen_mass", [1.1, 3.0])
def test_hmr_not_applied_to_tip4p(
    tip4p,
    water,
    hydrogen_mass,
):
    pytest.importorskip("openmm")

    system = tip4p.create_interchange(
        Topology.from_molecules(2 * [water]),
    ).to_openmm_system(
        hydrogen_mass=hydrogen_mass,
    )

    masses = [system.getParticleMass(particle_index)._value for particle_index in range(system.getNumParticles())]

    assert masses[0] == masses[3] == 15.99943
    assert masses[1] == masses[2] == masses[4] == masses[5] == 1.007947
    assert masses[6] == masses[7] == 0.0


@pytest.mark.parametrize("hydrogen_mass", [1.1, 3.0])
def test_hmr_with_ligand_virtual_sites(sage_with_bond_charge, hydrogen_mass):
    """Test that a single-molecule sigma hole example runs"""
    pytest.importorskip("openmm")
    import openmm
    import openmm.unit

    topology = MoleculeWithConformer.from_mapped_smiles(
        "[H:5][C:2]([H:6])([C:3]([H:7])([H:8])[Cl:4])[Cl:1]",
    ).to_topology()
    topology.box_vectors = Quantity([4, 4, 4], "nanometer")

    # should work fine with 4 fs, but be conservative and just use 3 fs
    simulation = sage_with_bond_charge.create_interchange(topology).to_openmm_simulation(
        integrator=openmm.VerletIntegrator(3.0 * openmm.unit.femtosecond),
        hydrogen_mass=hydrogen_mass,
    )

    system = simulation.system

    assert system.getParticleMass(0)._value == 35.4532
    assert system.getParticleMass(3)._value == 35.4532

    assert system.getParticleMass(1)._value == 12.01078 - 2 * (hydrogen_mass - 1.007947)
    assert system.getParticleMass(2)._value == 12.01078 - 2 * (hydrogen_mass - 1.007947)

    assert system.getParticleMass(4)._value == hydrogen_mass
    assert system.getParticleMass(5)._value == hydrogen_mass
    assert system.getParticleMass(6)._value == hydrogen_mass
    assert system.getParticleMass(7)._value == hydrogen_mass

    assert system.getParticleMass(8)._value == 0.0
    assert system.getParticleMass(9)._value == 0.0

    # make sure it can minimize and run briefly without crashing
    # this is a single ligand in vacuum with boring parameters
    simulation.minimizeEnergy(maxIterations=100)

    simulation.runForClockTime(10 * openmm.unit.second)
