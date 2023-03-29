from openff.interchange.interop.gromacs._import._import import GROMACSSystem
from openff.interchange.interop.gromacs._import._topology import (
    _create_topology_from_system,
)
from openff.interchange.tests import get_test_file_path


def test_complex():
    topology = _create_topology_from_system(
        GROMACSSystem.from_top(get_test_file_path("complex.top")),
    )

    assert topology.n_molecules == 2165
    assert topology.n_atoms == 6656

    assert topology.molecule(0).n_atoms == 144
    assert topology.molecule(1).n_atoms == 49
    assert topology.molecule(2).n_atoms == 1
    assert topology.molecule(15).n_atoms == 3
    assert topology.molecule(-1).n_atoms == 3

    assert topology.molecule(2).n_bonds == 0
    assert topology.molecule(15).n_bonds == 2
    assert topology.molecule(-1).n_bonds == 2
