import random

import parmed
import pytest
from openff.toolkit import unit

from openff.interchange.exceptions import UnsupportedExportError


@pytest.mark.parametrize("reversed", [False, True])
def test_hmr_basic(sage, reversed, ethanol, reversed_ethanol):
    hydrogen_mass = random.uniform(1.0, 4.0)

    molecule = reversed_ethanol if reversed else ethanol
    molecule.generate_conformers(n_conformers=1)

    topology = molecule.to_topology()

    interchange = sage.create_interchange(topology)

    interchange.to_gromacs(prefix="asdf", hydrogen_mass=hydrogen_mass)

    structure = parmed.load_file("asdf.top")

    expected_mass = sum([atom.mass for atom in topology.atoms]).m_as(unit.dalton)

    found_mass = sum([atom.mass for atom in structure.atoms])

    assert found_mass == pytest.approx(expected_mass)

    for particle_index, atom in enumerate(topology.atoms):

        if atom.atomic_number == 1:
            assert structure.atoms[particle_index].mass == pytest.approx(hydrogen_mass)


def test_hmr_not_applied_to_water(sage, water):
    # TODO: This should have different behavior for rigid and flexible water,
    #       but sage has tip3p (rigid) so it should always be skipped
    hydrogen_mass = 1.23

    interchange = sage.create_interchange(water.to_topology())

    interchange.to_gromacs(prefix="fff", hydrogen_mass=hydrogen_mass)

    structure = parmed.load_file("fff.top")

    expected_mass = sum([atom.mass for atom in water.atoms]).m_as(unit.dalton)

    found_mass = sum([atom.mass for atom in structure.atoms])

    assert found_mass == pytest.approx(expected_mass)

    for particle_index, atom in enumerate(interchange.topology.atoms):

        if atom.atomic_number == 1:
            assert structure.atoms[particle_index].mass == pytest.approx(1.007947)


def test_virtual_sites_unsupported(tip4p, water):
    with pytest.raises(UnsupportedExportError, match="with virtual sites"):
        tip4p.create_interchange(water.to_topology()).to_gromacs(
            prefix="no_vs",
            hydrogen_mass=2.0,
        )
