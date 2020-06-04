import pytest
import pint

from system import System, PotentialHandler, PotentialCollection, Topology, handler_conversion
from system.tests.base_test import BaseTest


u = pint.UnitRegistry()

class TestPotentialHandler(BaseTest):
    def test_handler_conversion(self, argon_ff):
        collection = handler_conversion(
            forcefield=argon_ff,
            potential_collection=PotentialCollection(
                parameters={
                    'vdW': PotentialHandler(name='vdW'),
                }
            ),
            handler_name='vdW',
        )

        assert collection['vdW']['[#18:1]'].parameters['sigma'] == 0.3 * u.nm

class TestPotentialCollection(BaseTest):
    def test_potential_collection_from_toolkit(self, argon_ff):
        ff = PotentialCollection.from_toolkit_forcefield(argon_ff)
        assert ff is not None


class TestTopology(BaseTest):
    def test_topology_from_toolkit(self, argon_top):
        top = Topology.from_toolkit_topology(argon_top)
        assert top is not None


class TestSystem(BaseTest):
    def test_constructor(self, argon_ff, argon_top):
        """Test the basic constructor"""
        System(
            topology=argon_top,
            potential_collection=argon_ff,
        )

    def test_run_typing(self, argon_ff, argon_top):
        """Test that run_typing properly stores the parameter id"""
        test_system = System(
            topology=argon_top,
            potential_collection=argon_ff,
        )

        test_system.run_typing(
            toolkit_forcefield=argon_ff,
            toolkit_topology=argon_top,
        )
