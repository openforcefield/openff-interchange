from typing import TYPE_CHECKING, Dict, List, Optional, Set, Type, Union

from openff.toolkit import ForceField, Molecule, Topology
from openff.toolkit.typing.engines.smirnoff.plugins import load_handler_plugins
from openff.units import Quantity
from packaging.version import Version

from openff.interchange import Interchange
from openff.interchange.components.toolkit import _check_electrostatics_handlers
from openff.interchange.exceptions import (
    MissingParameterHandlerError,
    SMIRNOFFHandlersNotImplementedError,
)
from openff.interchange.plugins import load_smirnoff_plugins
from openff.interchange.smirnoff._nonbonded import (
    SMIRNOFFElectrostaticsCollection,
    SMIRNOFFvdWCollection,
)
from openff.interchange.smirnoff._positions import _infer_positions
from openff.interchange.smirnoff._valence import (
    SMIRNOFFAngleCollection,
    SMIRNOFFBondCollection,
    SMIRNOFFConstraintCollection,
    SMIRNOFFImproperTorsionCollection,
    SMIRNOFFProperTorsionCollection,
)
from openff.interchange.smirnoff._virtual_sites import SMIRNOFFVirtualSiteCollection

if TYPE_CHECKING:
    from openff.toolkit.typing.engines.smirnoff import ParameterHandler

    from openff.interchange.smirnoff._base import SMIRNOFFCollection

_SUPPORTED_PARAMETER_HANDLERS: Set[str] = {
    "Constraints",
    "Bonds",
    "Angles",
    "ProperTorsions",
    "ImproperTorsions",
    "vdW",
    "Electrostatics",
    "LibraryCharges",
    "ChargeIncrementModel",
    "VirtualSites",
}

_PLUGIN_CLASS_MAPPING: Dict[
    Type["ParameterHandler"],
    Type["SMIRNOFFCollection"],
] = dict()

for collection_plugin in load_smirnoff_plugins():
    parameter_handlers: List[
        Type["ParameterHandler"]
    ] = collection_plugin.allowed_parameter_handlers()

    if len(parameter_handlers) != 1:
        raise RuntimeError

    if parameter_handlers[0] in load_handler_plugins():
        _SUPPORTED_PARAMETER_HANDLERS.add(parameter_handlers[0]._TAGNAME)
        _PLUGIN_CLASS_MAPPING[parameter_handlers[0]] = collection_plugin


def _check_supported_handlers(force_field: ForceField):
    unsupported = list()

    for handler_name in force_field.registered_parameter_handlers:
        if handler_name in {"ToolkitAM1BCC"}:
            continue
        if handler_name not in _SUPPORTED_PARAMETER_HANDLERS:
            unsupported.append(handler_name)

    if unsupported:
        raise SMIRNOFFHandlersNotImplementedError(
            f"SMIRNOFF section(s) not implemented in Interchange: {unsupported}",
        )


def _create_interchange(
    force_field: ForceField,
    topology: Union[Topology, List[Molecule]],
    box: Optional[Quantity] = None,
    positions: Optional[Quantity] = None,
    charge_from_molecules: Optional[List[Molecule]] = None,
    partial_bond_orders_from_molecules: Optional[List[Molecule]] = None,
    allow_nonintegral_charges: bool = False,
) -> Interchange:
    _check_supported_handlers(force_field)

    interchange = Interchange()

    _topology = Interchange.validate_topology(topology)

    interchange.positions = _infer_positions(_topology, positions)

    interchange.box = _topology.box_vectors if box is None else box

    _bonds(interchange, force_field, _topology, partial_bond_orders_from_molecules)
    _constraints(
        interchange,
        force_field,
        _topology,
        bonds=interchange.collections.get("Bonds", None),  # type: ignore[arg-type]
    )
    _angles(interchange, force_field, _topology)
    _propers(interchange, force_field, _topology, partial_bond_orders_from_molecules)
    _impropers(interchange, force_field, _topology)

    _vdw(interchange, force_field, _topology)
    _electrostatics(
        interchange,
        force_field,
        _topology,
        charge_from_molecules,
        allow_nonintegral_charges,
    )
    _virtual_sites(interchange, force_field, _topology)

    _plugins(interchange, force_field, _topology)

    interchange.topology = _topology

    return interchange


def _bonds(
    interchange: Interchange,
    force_field: ForceField,
    _topology: Topology,
    partial_bond_orders_from_molecules: Optional[List[Molecule]] = None,
):
    if "Bonds" not in force_field.registered_parameter_handlers:
        return

    if force_field["Bonds"].version == Version("0.3"):
        from openff.interchange.smirnoff._valence import _upconvert_bondhandler

        _upconvert_bondhandler(force_field["Bonds"])

    interchange.collections.update(
        {
            "Bonds": SMIRNOFFBondCollection.create(
                parameter_handler=force_field["Bonds"],
                topology=_topology,
                partial_bond_orders_from_molecules=partial_bond_orders_from_molecules,
            ),
        },
    )


def _constraints(
    interchange: Interchange,
    force_field: ForceField,
    topology: Topology,
    bonds: Optional[SMIRNOFFBondCollection] = None,
):
    interchange.collections.update(
        {
            "Constraints": SMIRNOFFConstraintCollection.create(
                parameter_handler=[
                    handler
                    for handler in [
                        force_field._parameter_handlers.get("Bonds", None),
                        force_field._parameter_handlers.get("Constraints", None),
                    ]
                    if handler is not None
                ],
                topology=topology,
                bonds=bonds,
            ),
        },
    )


def _angles(interchange, force_field, _topology):
    if "Angles" not in force_field.registered_parameter_handlers:
        return

    interchange.collections.update(
        {
            "Angles": SMIRNOFFAngleCollection.create(
                parameter_handler=force_field["Angles"],
                topology=_topology,
            ),
        },
    )


def _propers(
    interchange,
    force_field,
    _topology,
    partial_bond_orders_from_molecules=None,
):
    if "ProperTorsions" not in force_field.registered_parameter_handlers:
        return

    interchange.collections.update(
        {
            "ProperTorsions": SMIRNOFFProperTorsionCollection.create(
                parameter_handler=force_field["ProperTorsions"],
                topology=_topology,
                partial_bond_orders_from_molecules=partial_bond_orders_from_molecules,
            ),
        },
    )


def _impropers(interchange, force_field, _topology):
    if "ImproperTorsions" not in force_field.registered_parameter_handlers:
        return

    interchange.collections.update(
        {
            "ImproperTorsions": SMIRNOFFImproperTorsionCollection.create(
                parameter_handler=force_field["ImproperTorsions"],
                topology=_topology,
            ),
        },
    )


def _vdw(interchange: Interchange, force_field: ForceField, topology: Topology):
    if "vdW" not in force_field.registered_parameter_handlers:
        return

    interchange.collections.update(
        {
            "vdW": SMIRNOFFvdWCollection.create(
                parameter_handler=force_field["vdW"],
                topology=topology,
            ),
        },
    )


def _electrostatics(
    interchange: Interchange,
    force_field: ForceField,
    topology: Topology,
    charge_from_molecules: Optional[List[Molecule]] = None,
    allow_nonintegral_charges: bool = False,
):
    if "Electrostatics" not in force_field.registered_parameter_handlers:
        if _check_electrostatics_handlers(force_field):
            raise MissingParameterHandlerError(
                "Force field contains parameter handler(s) that may assign/modify "
                "partial charges, but no ElectrostaticsHandler was found.",
            )

        else:
            return

    interchange.collections.update(
        {
            "Electrostatics": SMIRNOFFElectrostaticsCollection.create(
                parameter_handler=[
                    handler
                    for handler in [
                        force_field._parameter_handlers.get(name, None)
                        for name in [
                            "Electrostatics",
                            "ChargeIncrementModel",
                            "ToolkitAM1BCC",
                            "LibraryCharges",
                        ]
                    ]
                    if handler is not None
                ],
                topology=topology,
                charge_from_molecules=charge_from_molecules,
                allow_nonintegral_charges=allow_nonintegral_charges,
            ),
        },
    )


def _virtual_sites(
    interchange: Interchange,
    force_field: ForceField,
    topology: Topology,
):
    if "VirtualSites" not in force_field.registered_parameter_handlers:
        return

    virtual_site_handler = SMIRNOFFVirtualSiteCollection()

    virtual_site_handler.exclusion_policy = force_field["VirtualSites"].exclusion_policy

    virtual_site_handler.store_matches(
        parameter_handler=force_field["VirtualSites"],
        topology=topology,
    )

    vdw: SMIRNOFFvdWCollection = interchange["vdW"]  # type: ignore[assignment]
    electrostatics: SMIRNOFFElectrostaticsCollection = interchange["Electrostatics"]  # type: ignore[assignment]

    virtual_site_handler.store_potentials(
        parameter_handler=force_field["VirtualSites"],
        vdw_handler=vdw,
        electrostatics_handler=electrostatics,
    )

    interchange.collections.update({"VirtualSites": virtual_site_handler})


def _plugins(
    interchange: Interchange,
    force_field: ForceField,
    topology: Topology,
):
    for handler_class, collection_class in _PLUGIN_CLASS_MAPPING.items():
        if handler_class._TAGNAME not in force_field.registered_parameter_handlers:
            continue

        collection = collection_class.create(
            parameter_handler=force_field[handler_class._TAGNAME],
            topology=topology,
        )

        interchange.collections.update(
            {
                handler_class._TAGNAME: collection,
            },
        )
