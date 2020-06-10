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
        """Fixture that loads an SMIRNOFF XML"""
        return ForceField(get_test_file_path('ar.offxml'))

    @pytest.fixture
    def argon_top(self):
        """Fixture that builds a simple arogon topology"""
        mol = Molecule.from_smiles('[#18]')
        mol.generate_conformers(n_conformers=1)

        return Topology.from_molecules(10 * [mol])

    @pytest.fixture
    def argon_coords(self, argon_top):
        return np.zeros(shape=(argon_top.n_topology_atoms, 3))

    @pytest.fixture
    def argon_box(self):
        return np.array([1, 1, 1])
