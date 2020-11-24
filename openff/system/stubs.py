"""
Monkeypatching external classes with custom functionality
"""
from typing import Optional, Union

from openforcefield.topology.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.typing.engines.smirnoff.parameters import (
    AngleHandler,
    BondHandler,
    ConstraintHandler,
    ElectrostaticsHandler,
    ProperTorsionHandler,
    vdWHandler,
)
from simtk import unit as omm_unit

from openff.system.components.smirnoff import (
    SMIRNOFFAngleHandler,
    SMIRNOFFBondHandler,
    SMIRNOFFConstraintHandler,
    SMIRNOFFElectrostaticsHandler,
    SMIRNOFFProperTorsionHandler,
    SMIRNOFFvdWHandler,
)
from openff.system.components.system import System
from openff.system.types import BaseArray


def to_openff_system(
    self,
    topology: Topology,
    box: Optional[Union[omm_unit.Quantity, BaseArray]] = None,
    **kwargs,
) -> System:
    """
    A method, patched onto ForceField, that creates a System object

    """
    sys_out = System()

    for parameter_handler, potential_handler in mapping.items():
        if parameter_handler._TAGNAME not in [
            "Constraints",
            "Bonds",
            "Angles",
            "ProperTorsions",
            "vdW",
        ]:
            # Once the SMIRNOFF spec is implemented, this should be an exception
            continue
        if parameter_handler._TAGNAME not in self.registered_parameter_handlers:
            continue
        handler = self[parameter_handler._TAGNAME].create_potential(topology=topology)
        sys_out.handlers.update({parameter_handler._TAGNAME: handler})

    if "Electrostatics" in self.registered_parameter_handlers:
        charges = self["Electrostatics"].create_potential(
            forcefield=self, topology=topology
        )
        sys_out.handlers.update({"Electrostatics": charges})

    if box is None:
        sys_out.box = sys_out.validate_box(topology.box_vectors)
    else:
        sys_out.box = sys_out.validate_box(box)

    sys_out.topology = topology

    return sys_out


def create_constraint_handler(
    self,
    topology: Topology,
    **kwargs,
) -> SMIRNOFFConstraintHandler:
    """
    A method, patched onto ConstraintHandler, that creates a corresponding SMIRNOFFConstraintHandler

    """
    handler = SMIRNOFFConstraintHandler()
    handler.store_matches(parameter_handler=self, topology=topology)
    handler.store_constraints(parameter_handler=self)

    return handler


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


def create_vdw_potential_handler(
    self,
    topology: Topology,
    **kwargs,
) -> SMIRNOFFvdWHandler:
    """
    A method, patched onto BondHandler, that creates a corresponding SMIRNOFFBondHandler
    """
    handler = SMIRNOFFvdWHandler(
        scale_13=self.scale13,
        scale_14=self.scale14,
        scale_15=self.scale15,
    )
    handler.store_matches(parameter_handler=self, topology=topology)
    handler.store_potentials(parameter_handler=self)

    return handler


def create_charges(
    self, forcefield: ForceField, topology: Topology
) -> SMIRNOFFElectrostaticsHandler:
    handler = SMIRNOFFElectrostaticsHandler(
        scale_13=self.scale13,
        scale_14=self.scale14,
        scale_15=self.scale15,
    )
    handler.store_charges(forcefield=forcefield, topology=topology)

    return handler


mapping = {
    ConstraintHandler: SMIRNOFFConstraintHandler,
    BondHandler: SMIRNOFFBondHandler,
    AngleHandler: SMIRNOFFAngleHandler,
    ProperTorsionHandler: SMIRNOFFProperTorsionHandler,
    vdWHandler: SMIRNOFFvdWHandler,
}

ConstraintHandler.create_potential = create_constraint_handler
BondHandler.create_potential = create_bond_potential_handler
AngleHandler.create_potential = create_angle_potential_handler
ProperTorsionHandler.create_potential = create_proper_torsion_potential_handler
vdWHandler.create_potential = create_vdw_potential_handler
ElectrostaticsHandler.create_potential = create_charges
ForceField.create_openff_system = to_openff_system
