import warnings

from openff.toolkit import ForceField, Molecule, Quantity, Topology
from openff.toolkit.typing.engines.smirnoff import AngleHandler, BondHandler, ParameterHandler, ProperTorsionHandler
from openff.toolkit.typing.engines.smirnoff.plugins import load_handler_plugins
from packaging.version import Version

from openff.interchange import Interchange
from openff.interchange.common._positions import _infer_positions
from openff.interchange.components.toolkit import _check_electrostatics_handlers
from openff.interchange.exceptions import (
    InvalidTopologyError,
    MissingParameterHandlerError,
    PresetChargesError,
    SMIRNOFFHandlersNotImplementedError,
)
from openff.interchange.plugins import load_smirnoff_plugins
from openff.interchange.smirnoff._base import SMIRNOFFCollection, _check_all_valence_terms_assigned
from openff.interchange.smirnoff._gbsa import SMIRNOFFGBSACollection
from openff.interchange.smirnoff._nonbonded import (
    SMIRNOFFElectrostaticsCollection,
    SMIRNOFFvdWCollection,
    _upconvert_vdw_handler,
)
from openff.interchange.smirnoff._valence import (
    SMIRNOFFAngleCollection,
    SMIRNOFFBondCollection,
    SMIRNOFFConstraintCollection,
    SMIRNOFFImproperTorsionCollection,
    SMIRNOFFProperTorsionCollection,
    _upconvert_bondhandler,
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
    for parameter_handler in collection_plugin.allowed_parameter_handlers():
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
        if handler_name in {"ToolkitAM1BCC", "NAGLCharges"}:
            continue
        if handler_name not in _SUPPORTED_PARAMETER_HANDLERS:
            unsupported.append(handler_name)

    if unsupported:
        raise SMIRNOFFHandlersNotImplementedError(
            f"SMIRNOFF section(s) not implemented in Interchange: {unsupported}",
        )


def validate_topology(value):
    """Validate a topology-like argument, spliced from a previous validator."""
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
    if "Bonds" in force_field.registered_parameter_handlers:
        assigned_bond_indices = {tuple(key.atom_indices) for key in interchange["Bonds"].key_map}

        # need to filter the list of _all_ constraints to those that are bond-like, since constraints
        # can also be used to freeze i-k atoms in angles and other things, and we don't want those to
        # be interpreted as over-assigned bond terms
        topological_bond_indices = {
            (
                interchange.topology.atom_index(bond.atom1),
                interchange.topology.atom_index(bond.atom2),
            )
            for bond in interchange.topology.bonds
        }
        all_assigned_constraint_indices = {tuple(key.atom_indices) for key in interchange["Constraints"].key_map}
        bond_like_assigned_constraint_indices = {
            constraint for constraint in all_assigned_constraint_indices if constraint in topological_bond_indices
        }

        _check_all_valence_terms_assigned(
            handler_class=BondHandler,
            topology=interchange.topology,
            assigned_atom_indices=assigned_bond_indices.union(bond_like_assigned_constraint_indices),
            valence_terms=interchange["Bonds"].valence_terms(interchange.topology),
        )

    _angles(interchange, force_field, interchange.topology)

    if "Angles" in force_field.registered_parameter_handlers:
        assigned_angle_indices = {tuple(key.atom_indices) for key in interchange["Angles"].key_map}

        # need to filter the list of angles whose i-k atoms are convered by constraints
        topological_angle_indices = {
            (
                interchange.topology.atom_index(angle[0]),
                interchange.topology.atom_index(angle[1]),
                interchange.topology.atom_index(angle[2]),
            )
            for angle in interchange.topology.angles
        }
        all_assigned_constraint_indices = {tuple(key.atom_indices) for key in interchange["Constraints"].key_map}
        angles_mimicked_by_constraints = {
            pair for pair in topological_angle_indices if tuple((pair[0], pair[2])) in all_assigned_constraint_indices
        }

        _check_all_valence_terms_assigned(
            handler_class=AngleHandler,
            topology=interchange.topology,
            assigned_atom_indices=assigned_angle_indices.union(angles_mimicked_by_constraints),
            valence_terms=interchange["Angles"].valence_terms(interchange.topology),
        )

    _propers(
        interchange,
        force_field,
        interchange.topology,
        partial_bond_orders_from_molecules,
    )
    _impropers(interchange, force_field, interchange.topology)

    for handler_name, handler_class in zip(
        ["ProperTorsions"],
        [ProperTorsionHandler],
    ):
        if handler_name in force_field.registered_parameter_handlers:
            _check_all_valence_terms_assigned(
                handler_class=handler_class,
                topology=interchange.topology,
                assigned_atom_indices={
                    *{tuple(key.atom_indices) for key in interchange[handler_name].key_map},
                },
                valence_terms=interchange[handler_name].valence_terms(interchange.topology),  # type: ignore[attr-defined]
            )

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
                            "NAGLCharges",
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


def _build_collection(
    collection_class: type[SMIRNOFFCollection],
    handler_classes: list[type[ParameterHandler]],
    interchange: Interchange,
    force_field: ForceField,
    topology: Topology,
):
    kwargs = {}

    # Always attach handlers
    if len(handler_classes) == 1:
        kwargs["parameter_handler"] = force_field[handler_classes[0]._TAGNAME]
    else:
        kwargs["parameter_handler"] = [force_field[cls._TAGNAME] for cls in handler_classes]

    kwargs["topology"] = topology

    # VirtualSite collections need vdW + Electrostatics collections, and we can identify
    # them by the allowed_vdw_parameter_handlers class method.
    if hasattr(collection_class, "allowed_vdw_parameter_handlers"):
        vdw_tagnames = [x._TAGNAME for x in collection_class.allowed_vdw_parameter_handlers()]

        if (n_vdw_tagnames := len(vdw_tagnames)) != 1:
            raise NotImplementedError(
                f"Collection {collection_class} requires {n_vdw_tagnames} vdW handlers, but only one is supported.",
            )

        kwargs["vdw_collection"] = interchange[vdw_tagnames[0]]
        kwargs["electrostatics_collection"] = interchange["Electrostatics"]

    return collection_class.create(**kwargs)


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

        collection = _build_collection(collection_class, handler_classes, interchange, force_field, topology)

        # No matter if this collection takes one or multiple handlers, key it by its own name
        interchange.collections[collection.type] = collection
