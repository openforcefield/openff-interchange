import random

import pytest
from openff.toolkit import unit

from openff.interchange._tests import MoleculeWithConformer
from openff.interchange.exceptions import UnsupportedExportError


def test_hmr_basic(sage):
    pytest.importorskip("openmm.unit")
    import openmm.unit

    hydrogen_mass = random.uniform(1.0, 4.0)
    topology = MoleculeWithConformer.from_mapped_smiles(
        "[H:4][C:1]([H:5])([H:6])[C:2]([H:7])([H:8])[O:3][H:9]",
    ).to_topology()

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


def test_virtual_sites_unsupported(tip4p, water):
    with pytest.raises(UnsupportedExportError, match="with virtual sites"):
        tip4p.create_interchange(water.to_topology()).to_openmm(hydrogen_mass=2.0)
