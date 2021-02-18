"""
Monkeypatching external classes with custom functionality
"""
import numpy as np
from openff.toolkit.topology.topology import Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.typing.engines.smirnoff.parameters import (
    AngleHandler,
    BondHandler,
    ConstraintHandler,
    ImproperTorsionHandler,
    ProperTorsionHandler,
    vdWHandler,
)

from openff.system.components.smirnoff import (
    ElectrostaticsMetaHandler,
    SMIRNOFFAngleHandler,
    SMIRNOFFBondHandler,
    SMIRNOFFChargeIncrementHandler,
    SMIRNOFFConstraintHandler,
    SMIRNOFFImproperTorsionHandler,
    SMIRNOFFLibraryChargeHandler,
    SMIRNOFFProperTorsionHandler,
    SMIRNOFFvdWHandler,
)
from openff.system.components.system import System


def to_openff_system(
    self,
    topology: Topology,
    box=None,
    **kwargs,
) -> System:
    """
    A method, patched onto ForceField, that creates a System object

    """
    sys_out = System()

    _check_supported_handlers(self)

    for parameter_handler in self.registered_parameter_handlers:
        if parameter_handler in {
            "Electrostatics",
            "ToolkitAM1BCC",
            "LibraryCharges",
            "ChargeIncrementModel",
            "Constraints",
        }:
            continue
        else:
            potential_handler = self[parameter_handler].create_potential(
                topology=topology
            )
        sys_out.handlers.update({parameter_handler: potential_handler})

    if "Constraints" in self.registered_parameter_handlers:
        constraint_handler = self["Constraints"].create_potential(
            topology=topology,
            bond_handler=sys_out.handlers["Bonds"]
            if "Bonds" in sys_out.handlers
            else None,
        )
        sys_out.handlers.update({"Constraints": constraint_handler})

    if "Electrostatics" in self.registered_parameter_handlers:
        electrostatics = ElectrostaticsMetaHandler(
            scale_13=self["Electrostatics"].scale13,
            scale_14=self["Electrostatics"].scale14,
            scale_15=self["Electrostatics"].scale15,
        )
        if "ToolkitAM1BCC" in self.registered_parameter_handlers:
            electrostatics.cache_charges(
                partial_charge_method="am1bcc", topology=topology
            )
            electrostatics.charges = electrostatics.cache["am1bcc"]

        if "LibraryCharges" in self.registered_parameter_handlers:
            library_charges = SMIRNOFFLibraryChargeHandler()
            library_charges.store_matches(self["LibraryCharges"], topology)
            library_charges.store_potentials(self["LibraryCharges"])
            sys_out.handlers.update({"LibraryCharges": electrostatics})  # type: ignore[dict-item]

            electrostatics.apply_library_charges(library_charges)

        if "ChargeIncrementModel" in self.registered_parameter_handlers:
            charge_increments = SMIRNOFFChargeIncrementHandler()
            charge_increments.store_matches(self["ChargeIncrementModel"], topology)
            charge_increments.store_potentials(self["ChargeIncrementModel"])
            sys_out.handlers.update({"LibraryCharges": electrostatics})  # type: ignore[dict-item]

            if charge_increments.partial_charge_method not in electrostatics.cache:
                electrostatics.cache_charges(
                    partial_charge_method=charge_increments.partial_charge_method,
                    topology=topology,
                )
            electrostatics.charges = electrostatics.cache[
                charge_increments.partial_charge_method
            ]

            electrostatics.apply_charge_increments(charge_increments)

        sys_out.handlers.update({"Electrostatics": electrostatics})  # type: ignore[dict-item]
    # if "Electrostatics" not in self.registered_parameter_handlers:
    #     if "LibraryCharges" in self.registered_parameter_handlers:
    #         library_charge_handler = SMIRNOFFLibraryChargeHandler()
    #         library_charge_handler.store_matches(
    #             parameter_handler=self["LibraryCharges"], topology=topology
    #         )
    #         library_charge_handler.store_potentials(
    #             parameter_handler=self["LibraryCharges"]
    #         )
    #         sys_out.handlers.update({"LibraryCharges": library_charge_handler})

    # `box` argument is only overriden if passed `None` and the input topology
    # has box vectors
    if box is None and topology.box_vectors is not None:
        from simtk import unit

        # getDefaultPeriodicBoxVectors() / unit.nanometer is a tuple
        sys_out.box = np.asarray(topology.box_vectors / unit.nanometer)
    else:
        sys_out.box = box

    sys_out.topology = topology

    return sys_out


def create_constraint_handler(
    self,
    topology: Topology,
    bond_handler=None,
    **kwargs,
) -> SMIRNOFFConstraintHandler:
    """
    A method, patched onto ConstraintHandler, that creates a corresponding SMIRNOFFConstraintHandler

    """
    handler = SMIRNOFFConstraintHandler()
    handler.store_matches(parameter_handler=self, topology=topology)
    handler.store_constraints(parameter_handler=self, bond_handler=bond_handler)

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
    A method, patched onto BondHandler, that creates a corresponding SMIRNOFFAngleHandler

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
    A method, patched onto ProperTorsionHandler, that creates a corresponding SMIRNOFFProperTorsionHandler
    """
    handler = SMIRNOFFProperTorsionHandler()
    handler.store_matches(parameter_handler=self, topology=topology)
    handler.store_potentials(parameter_handler=self)

    return handler


def create_improper_torsion_potential_handler(
    self,
    topology: Topology,
    **kwargs,
) -> SMIRNOFFImproperTorsionHandler:
    """
    A method, patched onto ImproperTorsionHandler, that creates a corresponding SMIRNOFFImproperTorsionHandler
    """
    handler = SMIRNOFFImproperTorsionHandler()
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


def _check_supported_handlers(forcefield: ForceField):
    supported_handlers = {
        "Constraints",
        "Bonds",
        "Angles",
        "ProperTorsions",
        "ImproperTorsions",
        "vdW",
        "Electrostatics",
        "LibraryCharges",
        "ChargeIncrementModel",
    }

    unsupported = list()
    for handler in forcefield.registered_parameter_handlers:
        if handler in {"ToolkitAM1BCC"}:
            continue
        if handler not in supported_handlers:
            unsupported.append(handler)

    if unsupported:
        from openff.system.exceptions import SMIRNOFFHandlersNotImplementedError

        raise SMIRNOFFHandlersNotImplementedError(unsupported)


mapping = {
    ConstraintHandler: SMIRNOFFConstraintHandler,
    BondHandler: SMIRNOFFBondHandler,
    AngleHandler: SMIRNOFFAngleHandler,
    ProperTorsionHandler: SMIRNOFFProperTorsionHandler,
    ImproperTorsionHandler: SMIRNOFFImproperTorsionHandler,
    vdWHandler: SMIRNOFFvdWHandler,
}

ConstraintHandler.create_potential = create_constraint_handler
BondHandler.create_potential = create_bond_potential_handler
AngleHandler.create_potential = create_angle_potential_handler
ProperTorsionHandler.create_potential = create_proper_torsion_potential_handler
ImproperTorsionHandler.create_potential = create_improper_torsion_potential_handler
vdWHandler.create_potential = create_vdw_potential_handler
ForceField.create_openff_system = to_openff_system
