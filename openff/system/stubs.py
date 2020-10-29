"""
Monkeypatching external classes with custom functionality
"""
from openforcefield.topology.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.typing.engines.smirnoff.parameters import (
    AngleHandler,
    BondHandler,
    ProperTorsionHandler,
    vdWHandler,
)

from openff.system.components.smirnoff import (
    SMIRNOFFAngleHandler,
    SMIRNOFFBondHandler,
    SMIRNOFFProperTorsionHandler,
    SMIRNOFFvdWHandler,
)
from openff.system.components.system import System


def to_openff_system(self, topology: Topology, **kwargs) -> System:
    """
    A method, patched onto ForceField, that creates a System object

    """
    system = System()

    for parameter_handler, potential_handler in mapping.items():
        if parameter_handler._TAGNAME not in [
            "Bonds",
            "Angles",
            "ProperTorsions",
            "vdW",
        ]:
            continue
        handler = self[parameter_handler._TAGNAME].create_potential(topology=topology)
        system.handlers.update({parameter_handler._TAGNAME: handler})

    return system


# These functions should all be reduced down to one, possibly with some creative
# use of functools
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


def create_vdw_potential_handler(
    self,
    topology: Topology,
    **kwargs,
) -> SMIRNOFFvdWHandler:
    """
    A method, patched onto BondHandler, that creates a corresponding SMIRNOFFBondHandler

    """
    handler = SMIRNOFFvdWHandler()
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


def create_proper_torsion_potential_handler(
    self,
    topology: Topology,
    **kwargs,
) -> SMIRNOFFProperTorsionHandler:
    """
    A method, patched onto BondHandler, that creates a corresponding SMIRNOFFBondHandler

    """
    handler = SMIRNOFFProperTorsionHandler()
    handler.store_matches(parameter_handler=self, topology=topology)
    handler.store_potentials(parameter_handler=self)

    return handler


# def create_potential_handler(
#     self,
#     topology: Topology,
#     handler_class: PotentialHandler,
#     **kwargs,
# ) -> PotentialHandler:
#     handler = handler_class()
#     handler.store_matches(parameter_handler=self, topology=topology)
#     handler.store_potentials(parameter_handler=self)
#     return functools.partial(handler_class, topology=topology)


mapping = {
    BondHandler: SMIRNOFFBondHandler,
    AngleHandler: SMIRNOFFAngleHandler,
    ProperTorsionHandler: SMIRNOFFProperTorsionHandler,
    vdWHandler: SMIRNOFFvdWHandler,
}

# for potential_handler, parameter_handler in mapping.items():
#   parameter_handler.create_potential = functools.partialmethod(
#       create_potential_handler, handler_class=potential_handler
#   )

# BondHandler.create_potential = functools.partialmethod(
#     create_potential_handler, handler_class=SMIRNOFFBondHandler
# )
# AngleHandler.create_potential = functools.partialmethod(
#     create_potential_handler, handler_class=SMIRNOFFAngleHandler
# )
# ProperTorsionHandler.create_potential = functools.partialmethod(
#     create_potential_handler, handler_class=SMIRNOFFProperTorsionHandler
# )
# vdWHandler.create_potential = functools.partialmethod(
#     create_potential_handler, handler_class=SMIRNOFFvdWHandler
# )

BondHandler.create_potential = create_bond_potential_handler
AngleHandler.create_potential = create_angle_potential_handler
vdWHandler.create_potential = create_vdw_potential_handler
ProperTorsionHandler.create_potential = create_proper_torsion_potential_handler
ForceField.create_openff_system = to_openff_system
