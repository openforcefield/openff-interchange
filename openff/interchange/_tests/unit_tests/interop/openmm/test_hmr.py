import random

import pytest
from openff.toolkit import Molecule, unit

from openff.interchange.exceptions import NegativeMassError, UnsupportedExportError


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

    expected_mass = sum([atom.mass for atom in topology.atoms]).m_as(unit.dalton)

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

    expected_mass = sum([atom.mass for atom in interchange.topology.atoms]).m_as(
        unit.dalton,
    )

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


def test_virtual_sites_unsupported(tip4p, water):
    with pytest.raises(UnsupportedExportError, match="with virtual sites"):
        tip4p.create_interchange(water.to_topology()).to_openmm(hydrogen_mass=2.0)
