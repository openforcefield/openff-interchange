import warnings

from openff.toolkit import ForceField, Molecule, Quantity, Topology
from openff.toolkit.typing.engines.smirnoff import ParameterHandler
from openff.toolkit.typing.engines.smirnoff.plugins import load_handler_plugins
from packaging.version import Version

from openff.interchange import Interchange
from openff.interchange.common._positions import _infer_positions
from openff.interchange.components.toolkit import _check_electrostatics_handlers
from openff.interchange.exceptions import (
    MissingParameterHandlerError,
    PresetChargesError,
    SMIRNOFFHandlersNotImplementedError,
)
from openff.interchange.plugins import load_smirnoff_plugins
from openff.interchange.smirnoff._base import SMIRNOFFCollection
from openff.interchange.smirnoff._gbsa import SMIRNOFFGBSACollection
from openff.interchange.smirnoff._nonbonded import (
    SMIRNOFFElectrostaticsCollection,
    SMIRNOFFvdWCollection,
)
from openff.interchange.smirnoff._valence import (
    SMIRNOFFAngleCollection,
    SMIRNOFFBondCollection,
    SMIRNOFFConstraintCollection,
    SMIRNOFFImproperTorsionCollection,
    SMIRNOFFProperTorsionCollection,
)
from openff.interchange.smirnoff._virtual_sites import SMIRNOFFVirtualSiteCollection
from openff.interchange.warnings import PresetChargesAndVirtualSitesWarning

_SUPPORTED_PARAMETER_HANDLERS: set[str] = {
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
    "GBSA",
}

_PLUGIN_CLASS_MAPPING: dict[
    type["ParameterHandler"],
    type["SMIRNOFFCollection"],
] = dict()

for collection_plugin in load_smirnoff_plugins():
    parameter_handlers: list[type["ParameterHandler"]] = collection_plugin.allowed_parameter_handlers()

    for parameter_handler in parameter_handlers:
        if parameter_handler in load_handler_plugins():
            _SUPPORTED_PARAMETER_HANDLERS.add(parameter_handler._TAGNAME)
            _PLUGIN_CLASS_MAPPING[parameter_handler] = collection_plugin
        else:
            raise ValueError(
                f"`SMIRNOFFCollection` plugin {collection_plugin} supports `ParameterHandler` "
                f"plugin {parameter_handler}, but it was not found in the `openff.toolkit.plugins` "
                "entry point. If this collection can use this handler but does not require it, "
                "please raise an issue on GitHub describing your use case.",
            )


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


def validate_topology(value):
    """Validate a topology-like argument, spliced from a previous validator."""
    from openff.interchange.exceptions import InvalidTopologyError

    if value is None:
        return None
    if isinstance(value, Topology):
        return Topology(other=value)
    elif isinstance(value, list):
        return Topology.from_molecules(value)
    else:
        raise InvalidTopologyError(
            "Could not process topology argument, expected openff.toolkit.Topology. "
            f"Found object of type {type(value)}.",
        )


def _preprocess_preset_charges(
    molecules_with_preset_charges: list[Molecule] | None,
) -> list[Molecule] | None:
    """
    Pre-process the molecules_with_preset_charges argument.

    If molecules_with_preset_charges is None, return None.

    If molecules_with_preset_charges is list[Molecule], ensure that

    1. The input is a list of Molecules
    2. Each molecule has assign partial charges
    3. Ensure no molecules are isomorphic with another in the list

    """
    if molecules_with_preset_charges is None:
        return None

    # This relies on Molecule.__eq__(), which may change and/or not have the same equality criteria
    # as we want here.
    # See https://github.com/openforcefield/openff-interchange/pull/1070#discussion_r1792728179
    molecule_set = {molecule for molecule in molecules_with_preset_charges}

    if len(molecule_set) != len(molecules_with_preset_charges):
        raise PresetChargesError(
            "All molecules in the molecules_with_preset_charges list must be isomorphically unique from each other",
        )

    for molecule in molecules_with_preset_charges:
        if molecule.partial_charges is None:
            raise PresetChargesError(
                "All molecules in the molecules_with_preset_charges list must have partial charges assigned.",
            )

    return molecules_with_preset_charges


def _create_interchange(
    force_field: ForceField,
    topology: Topology | list[Molecule],
    box: Quantity | None = None,
    positions: Quantity | None = None,
    molecules_with_preset_charges: list[Molecule] | None = None,
    partial_bond_orders_from_molecules: list[Molecule] | None = None,
    allow_nonintegral_charges: bool = False,
) -> Interchange:
    molecules_with_preset_charges = _preprocess_preset_charges(molecules_with_preset_charges)

    _check_supported_handlers(force_field)

    if molecules_with_preset_charges is not None and "VirtualSites" in force_field.registered_parameter_handlers:
        warnings.warn(
            "Preset charges were provided (via `charge_from_molecules`) alongside a force field that includes "
            "virtual site parameters. Note that virtual sites will be applied charges from the force field and "
            "cannot be given preset charges. Virtual sites may also affect the charges of their orientation "
            "atoms, even if those atoms are given preset charges.",
            PresetChargesAndVirtualSitesWarning,
        )

    # interchange = Interchange(topology=topology)
    # or maybe
    interchange = Interchange(topology=validate_topology(topology))

    interchange.positions = _infer_positions(interchange.topology, positions)

    interchange.box = interchange.topology.box_vectors if box is None else box

    _bonds(
        interchange,
        force_field,
        interchange.topology,
        partial_bond_orders_from_molecules,
    )
    _constraints(
        interchange,
        force_field,
        interchange.topology,
        bonds=interchange.collections.get("Bonds", None),  # type: ignore[arg-type]
    )
    _angles(interchange, force_field, interchange.topology)
    _propers(
        interchange,
        force_field,
        interchange.topology,
        partial_bond_orders_from_molecules,
    )
    _impropers(interchange, force_field, interchange.topology)

    _vdw(interchange, force_field, interchange.topology)
    _electrostatics(
        interchange,
        force_field,
        interchange.topology,
        molecules_with_preset_charges,
        allow_nonintegral_charges,
    )
    _plugins(interchange, force_field, interchange.topology)

    _virtual_sites(interchange, force_field, interchange.topology)

    _gbsa(interchange, force_field, interchange.topology)

    interchange.topology = interchange.topology

    return interchange


def _bonds(
    interchange: Interchange,
    force_field: ForceField,
    _topology: Topology,
    partial_bond_orders_from_molecules: list[Molecule] | None = None,
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
    bonds: SMIRNOFFBondCollection | None = None,
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
    from openff.interchange.smirnoff._nonbonded import _upconvert_vdw_handler

    if "vdW" not in force_field.registered_parameter_handlers:
        return

    # TODO: This modifies a user-supplied argument in-place, might consider
    # deepcopying it somewhere around `Interchange.from_x`

    _upconvert_vdw_handler(force_field["vdW"])

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
    molecules_with_preset_charges: list[Molecule] | None = None,
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
                molecules_with_preset_charges=molecules_with_preset_charges,
                allow_nonintegral_charges=allow_nonintegral_charges,
            ),
        },
    )


def _gbsa(
    interchange: Interchange,
    force_field: ForceField,
    _topology: Topology,
):
    if "GBSA" not in force_field.registered_parameter_handlers:
        return

    interchange.collections.update(
        {
            "GBSA": SMIRNOFFGBSACollection.create(
                parameter_handler=force_field["GBSA"],
                topology=_topology,
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

    try:
        vdw = interchange["vdW"]
    except LookupError:
        # There might not be a handler named "vdW" but there could be a plugin that
        # is directed to act as it
        for collection in interchange.collections.values():
            if collection.is_plugin:
                if collection.acts_as == "vdW":  # type: ignore[attr-defined]
                    vdw = collection  # type: ignore[assignment]
                    break
        else:
            vdw = None

    electrostatics: SMIRNOFFElectrostaticsCollection = interchange["Electrostatics"]  # type: ignore[assignment]

    virtual_site_handler.store_potentials(
        parameter_handler=force_field["VirtualSites"],
        vdw_collection=vdw,  # type: ignore[arg-type]
        electrostatics_collection=electrostatics,
    )

    interchange.collections.update({"VirtualSites": virtual_site_handler})


def _plugins(
    interchange: Interchange,
    force_field: ForceField,
    topology: Topology,
):
    for collection_class in _PLUGIN_CLASS_MAPPING.values():
        # Track the handlers (keys) that map to this collection (value)
        handler_classes = [
            handler for handler in _PLUGIN_CLASS_MAPPING if _PLUGIN_CLASS_MAPPING[handler] == collection_class
        ]

        if not all(
            [handler_class._TAGNAME in force_field.registered_parameter_handlers for handler_class in handler_classes],
        ):
            continue

        if len(handler_classes) == 0:
            continue

        if len(handler_classes) == 1:
            handler_class = handler_classes[0]
            try:
                collection = collection_class.create(
                    parameter_handler=force_field[handler_class._TAGNAME],
                    topology=topology,
                )
            except TypeError:
                tagnames = [x._TAGNAME for x in collection_class.allowed_parameter_handlers()]

                if len(tagnames) > 1:
                    raise NotImplementedError(
                        f"Collection {collection} requires multiple handlers, but only one was provided.",
                    )

                try:
                    collection = collection_class.create(  # type: ignore[call-arg]
                        parameter_handler=force_field[handler_class._TAGNAME],
                        topology=topology,
                        vdw_collection=interchange[tagnames[0]],
                        electrostatics_collection=interchange["Electrostatics"],
                    )
                except TypeError:
                    collection = collection_class.create(  # type: ignore[call-arg]
                        parameter_handler=force_field[handler_class._TAGNAME],
                        topology=topology,
                        vdw_collection=interchange[tagnames[0]],
                        electrostatics_collection=interchange["Electrostatics"],
                    )
        else:
            # If this collection takes multiple handlers, pass it a list. Consider making this type the default.
            handlers: list[ParameterHandler] = [
                force_field[handler_class._TAGNAME] for handler_class in _PLUGIN_CLASS_MAPPING.keys()
            ]

            collection = collection_class.create(
                parameter_handler=handlers,
                topology=topology,
            )

        # No matter if this collection takes one or multiple handlers, key it by its own name
        interchange.collections.update(
            {
                collection.type: collection,
            },
        )
