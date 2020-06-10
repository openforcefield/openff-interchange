import pytest
import pint
from pydantic import ValidationError

from ..system import System
from ..collections import PotentialHandler, PotentialCollection
from ..typing.smirnoff import add_handler, build_smirnoff_map, build_smirnoff_collection
from .base_test import BaseTest


u = pint.UnitRegistry()


class TestPotentialHandler(BaseTest):
    def test_handler_conversion(self, argon_ff):
        collection = add_handler(
            forcefield=argon_ff,
            potential_collection=PotentialCollection(
                parameters={
                    'vdW': PotentialHandler(name='vdW'),
                }
            ),
            handler_name='vdW',
        )

        assert collection['vdW']['[#18:1]'].parameters['sigma'] == 0.3 * u.nm


class TestSystem(BaseTest):

    @pytest.fixture
    def smirks_map(self, argon_ff, argon_top):
        return build_smirnoff_map(forcefield=argon_ff, topology=argon_top)

    @pytest.fixture
    def smirnoff_collection(self, argon_ff):
        return build_smirnoff_collection(argon_ff)

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
