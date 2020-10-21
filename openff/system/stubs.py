"""
Monkeypatching external classes with custom functionality
"""
from openforcefield.topology.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.typing.engines.smirnoff.parameters import BondHandler

from openff.system.components.smirnoff import SMIRNOFFBondHandler


class System:
    """
    A fake system meant only to demonstrate how `PotentialHandler`s are
    meant to be structured

    """


def to_openff_system(self, topology: Topology, **kwargs) -> System:
    """
    A method, patched onto ForceField, that creates a System object

    """
    system = System()
    system.handlers = dict()

    bonds = self["Bonds"].create_bond_potential_handler(topology=topology, **kwargs)
    system.handlers.update({"Bonds": bonds})

    return system


def create_bond_potential_handler(
    self, topology: Topology, **kwargs
) -> SMIRNOFFBondHandler:
    """
    A method, patched onto BondHandler, that creates a corresponding SMIRNOFFBondHandler

    """
    handler = SMIRNOFFBondHandler()
    handler.store_matches(parameter_handler=self, topology=topology)
    handler.store_potentials(parameter_handler=self)

    return handler


BondHandler.create_bond_potential_handler = create_bond_potential_handler
ForceField.create_openff_system = to_openff_system
