import numpy as np
import pytest
from openforcefield.topology.molecule import Molecule
from openforcefield.topology.topology import Topology

from openff.system.stubs import ForceField
from openff.system.utils import get_test_file_path

from .utils import top_from_smiles


class BaseTest:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()

    # TODO: group fixtures up as dicts, i.e. argon['forcefield'], argon['topology'], ...
    @pytest.fixture
    def argon_ff(self):
        """Fixture that loads an SMIRNOFF XML for argon"""
        return ForceField(get_test_file_path("argon.offxml"))

    @pytest.fixture
    def argon_top(self):
        """Fixture that builds a simple arogon topology"""
        return top_from_smiles("[#18]")

    @pytest.fixture
    def ammonia_ff(self):
        """Fixture that loads an SMIRNOFF XML for ammonia"""
        return ForceField(get_test_file_path("ammonia.offxml"))

    @pytest.fixture
    def ammonia_top(self):
        """Fixture that builds a simple ammonia topology"""
        mol = Molecule.from_smiles("N")

        return Topology.from_molecules(4 * [mol])

    @pytest.fixture
    def ethanol_top(self):
        """Fixture that builds a simple four ethanol topology"""
        return top_from_smiles("CCO", n_molecules=4)

    @pytest.fixture
    def cyclohexane_top(self):
        """Fixture that builds a simple cyclohexane topology"""
        return top_from_smiles("C1CCCCC1")

    @pytest.fixture
    def parsley(self):
        return ForceField("openff-1.0.0.offxml")

    @pytest.fixture
    def parsley_unconstrained(self):
        return ForceField("openff_unconstrained-1.0.0.offxml")

    @pytest.fixture
    def argon_coords(self, argon_top):
        return np.zeros(shape=(argon_top.n_topology_atoms, 3))

    @pytest.fixture
    def argon_box(self):
        return np.array([1, 1, 1])
