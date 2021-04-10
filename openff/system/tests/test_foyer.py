import glob
from pathlib import Path

import foyer
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.utils import get_data_file_path
from simtk import unit as omm_unit

from openff.system.components.foyer import from_foyer
from openff.system.tests.base_test import BaseTest
from openff.system.tests.energy_tests.gromacs import get_gromacs_energies
from openff.system.tests.energy_tests.openmm import get_openmm_energies
from openff.system.utils import get_test_files_dir_path


class TestFoyer(BaseTest):
    @pytest.fixture(scope="session")
    def oplsaa(self):
        return foyer.forcefields.load_OPLSAA()

    @pytest.fixture(scope="session")
    def oplsaa_system_ethanol(self, oplsaa):
        molecule = Molecule.from_file(get_data_file_path("molecules/ethanol.sdf"))
        molecule.name = "ETH"
        top = Topology.from_molecules(molecule)
        system = from_foyer(topology=top, ff=oplsaa)
        system.positions = molecule.conformers[0].value_in_unit(omm_unit.nanometer)
        system.box = [4, 4, 4]
        return system

    @pytest.fixture(scope="session")
    def openff_systems_from_foyer(self, oplsaa):
        foyer_mol_path = get_test_files_dir_path("foyer_test_molecules")
        foyer_mols = {
            Path(foyer_mol_path).stem: Topology.from_molecules(
                Molecule.from_file(foyer_mol_path)
            )
            for foyer_mol_path in glob.glob(f"{foyer_mol_path}/*.sdf")
        }

        foyer_systems = {}
        for name, topology in foyer_mols.items():
            topology.name = name
            # Currently Foyer cannot find atomtypes for the following molecules
            if name not in {"chlorine", "acetate"}:
                foyer_systems[name] = from_foyer(topology, oplsaa)

        return foyer_systems

    def test_handlers_exist(self, oplsaa_system_ethanol, openff_systems_from_foyer):
        for _, handler in oplsaa_system_ethanol.handlers.items():
            assert handler

        assert oplsaa_system_ethanol["vdW"].scale_14 == 0.5
        assert oplsaa_system_ethanol["Electrostatics"].scale_14 == 0.5

        for _, system in openff_systems_from_foyer.items():
            for _, handler in system.handlers.items():
                assert handler
            assert system["vdW"].scale_14 == 0.5
            assert system["Electrostatics"].scale_14 == 0.5

    def test_ethanol_energies(self, oplsaa_system_ethanol):
        gmx_energies = get_gromacs_energies(oplsaa_system_ethanol)
        omm_energies = get_openmm_energies(oplsaa_system_ethanol)

        gmx_energies.compare(
            omm_energies,
            custom_tolerances={"Nonbonded": 40.0 * omm_unit.kilojoule_per_mole},
        )
