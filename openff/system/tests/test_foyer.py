import foyer
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.utils import get_data_file_path
from simtk import unit as omm_unit

from openff.system.components.foyer import from_foyer
from openff.system.tests.base_test import BaseTest
from openff.system.tests.energy_tests.gromacs import get_gromacs_energies
from openff.system.tests.energy_tests.openmm import get_openmm_energies


class TestFoyer(BaseTest):
    @pytest.fixture(scope="session")
    def oplsaa_system_ethanol(self):
        molecule = Molecule.from_file(get_data_file_path("molecules/ethanol.sdf"))
        top = Topology.from_molecules(molecule)
        oplsaa = foyer.Forcefield(name="oplsaa")
        system = from_foyer(topology=top, ff=oplsaa)
        system.positions = molecule.conformers[0].value_in_unit(omm_unit.nanometer)
        system.box = [4, 4, 4]
        return system

    def test_handlers_exist(self, oplsaa_system_ethanol):
        for _, handler in oplsaa_system_ethanol.handlers.items():
            assert handler

    def test_ethanol_energies(self, oplsaa_system_ethanol):
        gmx_energies = get_gromacs_energies(oplsaa_system_ethanol)
        omm_energies = get_openmm_energies(oplsaa_system_ethanol)

        gmx_energies.compare(omm_energies)
