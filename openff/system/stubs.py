"""
Monkeypatching external classes with custom functionality
"""
from openforcefield.topology.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.typing.engines.smirnoff.parameters import AngleHandler, BondHandler

from openff.system.components.smirnoff import SMIRNOFFAngleHandler, SMIRNOFFBondHandler


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
    angles = self["Angles"].create_angle_potential_handler(topology=topology, **kwargs)
    system.handlers.update({"Bonds": bonds})
    system.handlers.update({"Angles": angles})

    return system


def create_bond_potential_handler(
    self,
    topology: Topology,
    **kwargs,
) -> SMIRNOFFBondHandler:
    """
    A method, patched onto BondHandler, that creates a corresponding SMIRNOFFBondHandler

    """
    handler = SMIRNOFFBondHandler()
    handler.store_matches(parameter_handler=self, topology=topology)
    handler.store_potentials(parameter_handler=self)

    return handler


def create_angle_potential_handler(
    self,
    topology: Topology,
    **kwargs,
) -> SMIRNOFFAngleHandler:
    """
    A method, patched onto BondHandler, that creates a corresponding SMIRNOFFBondHandler

    """
    handler = SMIRNOFFAngleHandler()
    handler.store_matches(parameter_handler=self, topology=topology)
    handler.store_potentials(parameter_handler=self)

    return handler


BondHandler.create_bond_potential_handler = create_bond_potential_handler
AngleHandler.create_angle_potential_handler = create_angle_potential_handler
ForceField.create_openff_system = to_openff_system
