import foyer
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.utils import get_data_file_path

from openff.system.components.foyer import from_foyer
from openff.system.tests.base_test import BaseTest


class TestFoyer(BaseTest):
    @pytest.fixture(scope="session")
    def oplsaa_system_ethanol(self):
        molecule = Molecule.from_file(get_data_file_path("molecules/ethanol.sdf"))
        top = Topology.from_molecules(molecule)
        oplsaa = foyer.Forcefield(name="oplsaa")
        system = from_foyer(topology=top, ff=oplsaa)
        return system

    def test_handlers(self, oplsaa_system_ethanol):
        for _, handler in oplsaa_system_ethanol.handlers.items():
            assert handler
