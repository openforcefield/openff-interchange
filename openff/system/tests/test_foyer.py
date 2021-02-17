import pytest
import parmed as pmd

from openff.system.components.foyer import from_foyer
from openff.system.tests.base_test import BaseTest


class TestFoyer(BaseTest):

    @pytest.fixture(scope='session')
    def oplsaa_system_ethane(self):
        import foyer  # type: ignore
        from foyer.tests.utils import get_fn  # type: ignore
        oplsaa = foyer.Forcefield(name='oplsaa')
        pmd_ethane = pmd.load_file(get_fn('ethane.mol2'), structure=True)
        system = from_foyer(
            structure=pmd_ethane,
            ff=oplsaa
        )
        return system

    def test_from_foyer_system_atom_handlers(self, oplsaa_system_ethane):
        assert oplsaa_system_ethane.handlers.get('FoyerAtomHandler')

    def test_from_foyer_system_bond_handlers(self, oplsaa_system_ethane):
        assert oplsaa_system_ethane.handlers.get('FoyerBondHandler')
