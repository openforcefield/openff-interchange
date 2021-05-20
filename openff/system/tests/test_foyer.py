import mdtraj as md
import pytest
from openff.toolkit.topology import Molecule
from openff.toolkit.utils import get_data_file_path
from openff.units import unit
from openff.utilities.testing import skip_if_missing
from openff.utilities.utilities import has_package
from simtk import unit as simtk_unit

from openff.system.components.mdtraj import OFFBioTop
from openff.system.tests.base_test import BaseTest
from openff.system.tests.energy_tests.openmm import get_openmm_energies

if has_package("foyer"):
    import foyer

    from openff.system.components.foyer import from_foyer


@skip_if_missing("foyer")
class TestFoyer(BaseTest):
    @pytest.fixture(scope="session")
    def oplsaa_system_ethanol(self):
        molecule = Molecule.from_file(get_data_file_path("molecules/ethanol.sdf"))
        molecule.name = "ETH"
        top = OFFBioTop.from_molecules(molecule)
        top.mdtop = md.Topology.from_openmm(top.to_openmm())
        oplsaa = foyer.Forcefield(name="oplsaa")
        system = from_foyer(topology=top, ff=oplsaa)
        system.positions = molecule.conformers[0].value_in_unit(simtk_unit.nanometer)
        system.box = [4, 4, 4]
        return system

    def test_handlers_exist(self, oplsaa_system_ethanol):
        for _, handler in oplsaa_system_ethanol.handlers.items():
            assert handler

        assert oplsaa_system_ethanol["vdW"].scale_14 == 0.5
        assert oplsaa_system_ethanol["Electrostatics"].scale_14 == 0.5

    @skip_if_missing("gromacs")
    @pytest.mark.slow
    def test_ethanol_energies(self, oplsaa_system_ethanol):
        from openff.system.tests.energy_tests.gromacs import get_gromacs_energies

        gmx_energies = get_gromacs_energies(oplsaa_system_ethanol)
        omm_energies = get_openmm_energies(oplsaa_system_ethanol)

        gmx_energies.compare(
            omm_energies,
            custom_tolerances={
                "vdW": 12.0 * unit.kilojoule / unit.mole,
                "Electrostatics": 12.0 * unit.kilojoule / unit.mole,
            },
        )
