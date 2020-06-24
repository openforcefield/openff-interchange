import pytest
from pydantic import ValidationError

from ..system import System, ProtoSystem
from ..typing.smirnoff import *
from .. import unit
from .base_test import BaseTest


class TestPotentialEnergyTerm(BaseTest):
    def test_term_conversion(self, argon_ff, argon_top):
        term = SMIRNOFFvdWTerm.build_from_toolkit_data(
            # TODO: name shouldn't be required here, something about the inheritance is dirty
            name='vdW',
            forcefield=argon_ff,
            topology=argon_top,
        )

        assert term.potentials['[#18:1]'].parameters['sigma'] == 0.3 * unit.nm


class TestSystem(BaseTest):

    @pytest.fixture
    def slot_smirks_map(self, argon_ff, argon_top):
        return build_slot_smirks_map(forcefield=argon_ff, topology=argon_top)

    @pytest.fixture
    def smirks_potential_map(self, argon_ff, slot_smirks_map):
        return build_smirks_potential_map(forcefield=argon_ff, smirks_map=slot_smirks_map)

    @pytest.fixture
    def term_collection(self, argon_ff, argon_top):
        return SMIRNOFFTermCollection.from_toolkit_data(toolkit_forcefield=argon_ff, toolkit_topology=argon_top)

    def test_constructor(self, argon_ff, argon_top, argon_coords, argon_box, slot_smirks_map, smirks_potential_map, term_collection):
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
            term_collection=term_collection,
            slot_smirks_map=slot_smirks_map,
            smirks_potential_map=smirks_potential_map,
            positions=argon_coords,
            box=argon_box,
        )

        with pytest.raises(ValidationError):
            System(
                term_collection=term_collection,
                potential_map=smirks_potential_map,
                positions=argon_coords,
                box=argon_box,
            )

        with pytest.raises(ValidationError):
            System(
                topology=argon_top,
                term_collection=term_collection,
                positions=argon_coords,
                box=argon_box,
            )

        with pytest.raises(ValidationError):
            System(
                topology=argon_top,
                potential_map=smirks_potential_map,
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

        assert test_system.slot_smirks_map is not None
        assert test_system.smirks_potential_map is not None


class TestProtoSystem(BaseTest):

    @pytest.mark.parametrize('values,units', [
        ([4, 4, 4], 'nm'),
        ([60, 60, 60], 'angstrom'),
        ([[4, 0, 0], [0, 4, 0], [0, 0, 4]], 'nm'),
    ])
    def test_valiate_box(self, values, units):
        box = ProtoSystem.validate_box(unit.Quantity(values, units=units))

        assert box.shape == (3, 3)
        assert box.units == unit.Unit(units)

    @pytest.mark.parametrize('values,units', [
        (3 * [4], 'hour'),
        (3 * [4], 'acre'),
        (2 * [4], 'nm'),
        (4 * [4], 'nm'),
    ])
    def test_validate_box_bad(self, values, units):
        with pytest.raises(TypeError):
            ProtoSystem.validate_box(values, units=units)

