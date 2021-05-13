"""
Monkeypatching external classes with custom functionality
"""
import mdtraj as md
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

from openff.system.components.mdtraj import OFFBioTop
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
from openff.system.exceptions import InvalidTopologyError

_MAPPING = {
    ConstraintHandler: SMIRNOFFConstraintHandler,
    BondHandler: SMIRNOFFBondHandler,
    AngleHandler: SMIRNOFFAngleHandler,
    ProperTorsionHandler: SMIRNOFFProperTorsionHandler,
    ImproperTorsionHandler: SMIRNOFFImproperTorsionHandler,
    vdWHandler: SMIRNOFFvdWHandler,
}


def to_openff_system(
    self,
    topology: OFFBioTop,
    box=None,
    **kwargs,
) -> System:
    """
    A method, patched onto ForceField, that creates a System object

    """
    sys_out = System()

    _check_supported_handlers(self)

    if isinstance(topology, OFFBioTop):
        sys_out.topology = topology
    elif isinstance(topology, Topology):
        sys_out.topology = OFFBioTop(topology)
        sys_out.topology.mdtop = md.Topology.from_openmm(topology.to_openmm())
    else:
        raise InvalidTopologyError(
            "Could not process topology argument, expected Topology or OFFBioTop. "
            f"Found object of type {type(topology)}."
        )

    for parameter_handler_name in self.registered_parameter_handlers:
        if parameter_handler_name in {
            "Electrostatics",
            "ToolkitAM1BCC",
            "LibraryCharges",
            "ChargeIncrementModel",
            "Constraints",
        }:
            continue
        elif parameter_handler_name == "Bonds":
            if "Constraints" in self.registered_parameter_handlers:
                constraint_handler = self["Constraints"]
            else:
                constraint_handler = None
            potential_handler, constraints = SMIRNOFFBondHandler.from_toolkit(
                bond_handler=self["Bonds"],
                topology=topology,
                constraint_handler=constraint_handler,
            )
            sys_out.handlers.update({"Bonds": potential_handler})
            if constraint_handler is not None:
                sys_out.handlers.update({"Constraints": constraints})
        elif parameter_handler_name in {
            "Angles",
            "ProperTorsions",
            "ImproperTorsions",
        }:
            parameter_handler = self[parameter_handler_name]
            POTENTIAL_HANDLER_CLASS = _MAPPING[parameter_handler.__class__]
            potential_handler = POTENTIAL_HANDLER_CLASS.from_toolkit(  # type: ignore
                parameter_handler=parameter_handler, topology=topology
            )
            sys_out.handlers.update({parameter_handler_name: potential_handler})
        elif parameter_handler_name == "vdW":
            potential_handler = SMIRNOFFvdWHandler._from_toolkit(  # type: ignore[assignment]
                parameter_handler=self["vdW"], topology=topology
            )
            sys_out.handlers.update({parameter_handler_name: potential_handler})
        else:
            potential_handler = self[parameter_handler_name].create_potential(
                topology=topology
            )
            sys_out.handlers.update({parameter_handler_name: potential_handler})

    if "Electrostatics" in self.registered_parameter_handlers:
        electrostatics = ElectrostaticsMetaHandler(
            scale_13=self["Electrostatics"].scale13,
            scale_14=self["Electrostatics"].scale14,
            scale_15=self["Electrostatics"].scale15,
            method=self["Electrostatics"].method.lower(),
            cutoff=self["Electrostatics"].cutoff,
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
        sys_out.box = topology.box_vectors
    else:
        sys_out.box = box

    return sys_out


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


ForceField.create_openff_system = to_openff_system
