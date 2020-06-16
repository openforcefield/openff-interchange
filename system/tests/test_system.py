import pytest
import pint
from pydantic import ValidationError

from ..system import System
from ..collections import *
from ..typing.smirnoff import *
from .base_test import BaseTest


u = pint.UnitRegistry()


class TestPotentialEnergyTerm(BaseTest):
    def test_term_conversion(self, argon_ff, argon_top):
        term = SMIRNOFFvdWTerm.build_from_toolkit_data(
            # TODO: name shouldn't be required here, something about the inheritance is dirty
            name='vdW',
            forcefield=argon_ff,
            topology=argon_top,
        )

        assert term.potentials['[#18:1]'].parameters['sigma'] == 0.3 * u.nm


class TestSystem(BaseTest):

    @pytest.fixture
    def smirks_map(self, argon_ff, argon_top):
        return build_slot_smirks_map(forcefield=argon_ff, topology=argon_top)

    @pytest.fixture
    def smirnoff_collection(self, argon_ff, argon_top):
        return SMIRNOFFTermCollection.from_toolkit_data(toolkit_forcefield=argon_ff, toolkit_topology=argon_top)

    def test_constructor(self, argon_ff, argon_top, argon_coords, argon_box, smirks_map, smirnoff_collection):
        """
        Test the basic constructor behavior from SMIRNOFF

        TODO: Check that the instances are reasonable, not just don't error
        """
        System(
            topology=argon_top,
            forcefield=argon_ff,
            positions=argon_coords,
            box=argon_box,
        )

        System(
            topology=argon_top,
            potential_collection=smirnoff_collection,
            potential_map=smirks_map,
            positions=argon_coords,
            box=argon_box,
        )

        with pytest.raises(ValidationError):
            System(
                potential_collection=smirnoff_collection,
                potential_map=smirks_map,
                positions=argon_coords,
                box=argon_box,
            )

        with pytest.raises(ValidationError):
            System(
                topology=argon_top,
                potential_collection=smirnoff_collection,
                positions=argon_coords,
                box=argon_box,
            )

        with pytest.raises(ValidationError):
            System(
                topology=argon_top,
                potential_map=smirks_map,
                positions=argon_coords,
                box=argon_box,
            )

    def test_automatic_typing(self, argon_ff, argon_top, argon_coords, argon_box):
        """
        Test that, when only forcefield is provided, typing dicts are built automatically.

        # TODO: Make sure the results are reasonble, not just existent.
        """
        test_system = System(
            topology=argon_top,
            forcefield=argon_ff,
            positions=argon_coords,
            box=argon_box,
        )

        assert test_system.potential_collection is not None
        assert test_system.potential_map is not None
