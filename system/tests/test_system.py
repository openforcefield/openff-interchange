import pytest

from system import System, PotentialCollection, Topology
from system.tests.base_test import BaseTest


class TestForceField(BaseTest):
    def test_forcefield_from_toolkit(self, argon_ff):
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
            topology=Topology.from_toolkit_topology(argon_top),
            forcefield=PotentialCollection.from_toolkit_forcefield(argon_ff),
        )

    def test_constructor_toolkit(self, argon_ff, argon_top):
        """Test the basic constructor directly from toolkit objects"""
        System(
            topology=argon_top,
            forcefield=argon_ff,
        )

    def test_run_typing(self, argon_ff, argon_top):
        """Test that run_typing properly stores the parameter id"""
        test_system = System(
            topology=argon_top,
            forcefield=argon_ff,
        )

        test_system.run_typing(
            toolkit_forcefield=argon_ff,
            toolkit_topology=argon_top,
        )

        for atom in test_system.topology.atoms:
            assert atom.parameter_id == 'n1'
