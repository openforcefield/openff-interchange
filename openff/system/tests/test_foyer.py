import pytest

from openff.system.components.foyer import from_foyer
from openff.system.tests.base_test import BaseTest


class TestFoyer(BaseTest):
    @pytest.fixture(scope="session")
    def oplsaa_system_ethane(self):
        import foyer
        from mbuild.conversion import to_parmed
        from mbuild.lib.molecules import Ethane

        oplsaa = foyer.Forcefield(name="oplsaa")
        pmd_ethane = to_parmed(Ethane())
        system = from_foyer(structure=pmd_ethane, ff=oplsaa)
        return system

    def test_from_foyer_system_atom_handlers(self, oplsaa_system_ethane):
        vdw_handler = oplsaa_system_ethane.handlers.get("FoyerVDWHandler")
        assert len(vdw_handler.potentials) == 2

    def test_from_foyer_system_bond_handlers(self, oplsaa_system_ethane):
        bond_handler = oplsaa_system_ethane.handlers.get("FoyerBondHandler")
        assert len(bond_handler.potentials) == 2
