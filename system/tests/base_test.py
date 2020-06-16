import pytest
import numpy as np

from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.topology.molecule import Molecule
from openforcefield.topology.topology import Topology

from system.utils import get_test_file_path


class BaseTest:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()

    @pytest.fixture
    def argon_ff(self):
        """Fixture that loads an SMIRNOFF XML for argon"""
        return ForceField(get_test_file_path('argon.offxml'))

    @pytest.fixture
    def argon_top(self):
        """Fixture that builds a simple arogon topology"""
        mol = Molecule.from_smiles('[#18]')
        mol.generate_conformers(n_conformers=1)

        return Topology.from_molecules(10 * [mol])

    @pytest.fixture
    def ammonia_top(self):
        """Fixture that builds a simple ammonia topology"""
        mol = Molecule.from_smiles('N')
        mol.generate_conformers(n_conformers=1)

        return Topology.from_molecules(10 * [mol])

    @pytest.fixture
    def argon_coords(self, argon_top):
        return np.zeros(shape=(argon_top.n_topology_atoms, 3))

    @pytest.fixture
    def argon_box(self):
        return np.array([1, 1, 1])

    @pytest.fixture
    def ammonia_ff(self):
        """Fixture that loads an SMIRNOFF XML for ammonia"""
        return ForceField(get_test_file_path('ammonia.offxml'))
