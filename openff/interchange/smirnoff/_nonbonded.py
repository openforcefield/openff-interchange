import copy
import functools
import logging
import warnings
from collections.abc import Iterable
from typing import Literal, Union

import numpy
from openff.toolkit import Molecule, Quantity, Topology, unit
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ChargeIncrementModelHandler,
    ElectrostaticsHandler,
    LibraryChargeHandler,
    ParameterHandler,
    ToolkitAM1BCCHandler,
    vdWHandler,
)
from pydantic import Field, PrivateAttr, computed_field
from typing_extensions import Self

from openff.interchange._annotations import _ElementaryChargeQuantity
from openff.interchange.common._nonbonded import (
    ElectrostaticsCollection,
    _NonbondedCollection,
    vdWCollection,
)
from openff.interchange.components.potentials import Potential
from openff.interchange.constants import _PME
from openff.interchange.exceptions import (
    InvalidParameterHandlerError,
    MissingPartialChargesError,
    NonIntegralMoleculeChargeError,
    SMIRNOFFVersionNotSupportedError,
)
from openff.interchange.models import (
    ChargeIncrementTopologyKey,
    ChargeModelTopologyKey,
    LibraryChargeTopologyKey,
    PotentialKey,
    SingleAtomChargeTopologyKey,
    TopologyKey,
    VirtualSiteKey,
)
from openff.interchange.smirnoff._base import SMIRNOFFCollection
from openff.interchange.warnings import ForceFieldModificationWarning

logger = logging.getLogger(__name__)

ElectrostaticsHandlerType = Union[
    ElectrostaticsHandler,
    ToolkitAM1BCCHandler,
    ChargeIncrementModelHandler,
    LibraryChargeHandler,
]

_ZERO_CHARGE = Quantity(0.0, "elementary_charge")


@functools.lru_cache(None)
def _add_charges(
    charge1: float,
    charge2: float,
) -> "Quantity":
    """Add two charges together."""
    return Quantity(charge1 + charge2, "elementary_charge")


def _upconvert_vdw_handler(vdw_handler: vdWHandler):
    """Given a vdW with version 0.3 or 0.4, up-convert to 0.4 or short-circuit if already 0.4."""
    from packaging.version import Version

    if vdw_handler.version >= Version("0.4"):
        return

    elif vdw_handler.version == Version("0.3"):
        warnings.warn(
            "Automatically up-converting vdWHandler from version 0.3 to 0.4. Consider manually upgrading "
            "this vdW (or <vdW> section in an OFFXML file) to 0.4 or newer. For more details, "
            "see https://openforcefield.github.io/standards/standards/smirnoff/#vdw.",
            ForceFieldModificationWarning,
        )

        if vdw_handler.method != "cutoff":
            raise NotImplementedError(
                "Up-converting vdWHandler with method != 'cutoff' not yet supported.",
            )

        vdw_handler.version = Version("0.4")
        vdw_handler.periodic_method = "cutoff"
        vdw_handler.nonbonded_method = "no-cutoff"

        delattr(vdw_handler, "method")


def _downconvert_vdw_handler(vdw_handler: vdWHandler):
    """Given a vdWHandler with version 0.4, down-convert to 0.3."""
    from packaging.version import Version

    if vdw_handler.version == Version("0.3"):
        return

    elif vdw_handler.version == Version("0.4"):
        warnings.warn(
            "Automatically down-converting vdWHandler from version 0.4 to 0.3. In the future, this "
            "down-conversion will not happen and version 0.3 will not be supported.",
            ForceFieldModificationWarning,
        )

        if (vdw_handler.periodic_method != "cutoff") or (vdw_handler.nonperiodic_method != "no-cutoff"):
            raise NotImplementedError("Down-converting vdWHandler failed.")

        vdw_handler.version = Version("0.3")
        vdw_handler.method = "cutoff"

        try:
            delattr(vdw_handler, "periodic_method")
            delattr(vdw_handler, "nonperiodic_method")
        except AttributeError:
            # If they cannot be deleted, just leave them hanging around. Not ideal!
            # https://github.com/openforcefield/openff-toolkit/issues/1680
            pass


def library_charge_from_molecule(
    molecule: "Molecule",
) -> LibraryChargeHandler.LibraryChargeType:
    """Given an OpenFF Molecule with charges, generate a corresponding LibraryChargeType."""
    if molecule.partial_charges is None:
        raise MissingPartialChargesError

    smirks = molecule.to_smiles(mapped=True)
    charges = molecule.partial_charges

    library_charge_type = LibraryChargeHandler.LibraryChargeType(
        smirks=smirks,
        charge=charges,
    )

    return library_charge_type


class _SMIRNOFFNonbondedCollection(SMIRNOFFCollection, _NonbondedCollection):
    """Base class for handlers storing non-bonded potentials produced by SMIRNOFF force fields."""


class SMIRNOFFvdWCollection(vdWCollection, SMIRNOFFCollection):
    """Handler storing vdW potentials as produced by a SMIRNOFF force field."""

    @classmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        return [vdWHandler]

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return a list of supported parameter attributes."""
        return ["smirks", "id", "sigma", "epsilon", "rmin_half"]

    @classmethod
    def default_parameter_values(cls) -> Iterable[float]:
        """Per-particle parameter values passed to Force.addParticle()."""
        return [1.0, 0.0]

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a list of names of parameters included in each potential in this colletion."""
        return ["sigma", "epsilon"]

    def store_potentials(self, parameter_handler: vdWHandler) -> None:
        """
        Populate self.potentials with key-val pairs of [TopologyKey, PotentialKey].

        """
        self.periodic_method = parameter_handler.periodic_method.lower().replace(
            "ewald3d",
            "pme",
        )
        self.nonperiodic_method = parameter_handler.nonperiodic_method.lower()
        self.cutoff = parameter_handler.cutoff

        for potential_key in self.key_map.values():
            smirks = potential_key.id
            force_field_parameters = parameter_handler.parameters[smirks]

            potential = Potential(
                parameters={
                    parameter: getattr(force_field_parameters, parameter) for parameter in self.potential_parameters()
                },
            )

            self.potentials[potential_key] = potential

    @classmethod
    def create(
        cls,
        parameter_handler: vdWHandler,
        topology: Topology,
    ) -> Self:
        """
        Create a SMIRNOFFvdWCollection from toolkit data.

        """
        if isinstance(parameter_handler, list):
            parameter_handlers = parameter_handler
        else:
            parameter_handlers = [parameter_handler]

        for parameter_handler in parameter_handlers:
            if type(parameter_handler) not in cls.allowed_parameter_handlers():
                raise InvalidParameterHandlerError(
                    f"Found parameter handler type {type(parameter_handler)}, which is not "
                    f"supported by potential type {type(cls)}",
                )

        handler = cls(
            scale_12=parameter_handler.scale12,
            scale_13=parameter_handler.scale13,
            scale_14=parameter_handler.scale14,
            scale_15=parameter_handler.scale15,
            cutoff=parameter_handler.cutoff,
            mixing_rule=parameter_handler.combining_rules.lower(),
            periodic_method=parameter_handler.periodic_method.lower().replace(
                "ewald3d",
                "pme",
            ),
            nonperiodic_method=parameter_handler.nonperiodic_method.lower(),
            switch_width=parameter_handler.switch_width,
        )
        handler.store_matches(parameter_handler=parameter_handler, topology=topology)
        handler.store_potentials(parameter_handler=parameter_handler)

        return handler

    @classmethod
    def parameter_handler_precedence(cls) -> list[str]:
        """
        Return the order in which parameter handlers take precedence when computing charges.
        """
        return ["vdw", "VirtualSites"]


class SMIRNOFFElectrostaticsCollection(ElectrostaticsCollection, SMIRNOFFCollection):
    """
    A handler which stores any electrostatic parameters applied to a topology.

    This handler is responsible for grouping together

    * global settings for the electrostatic interactions such as the cutoff distance
      and the intramolecular scale factors.
    * partial charges which have been assigned by a ``ToolkitAM1BCC``,
      ``LibraryCharges``, or a ``ChargeIncrementModel`` parameter
      handler.
    * charge corrections applied by a ``ChargeIncrementHandler``

    rather than having each in their own handler.
    """

    periodic_potential: Literal[
        "Ewald3D-ConductingBoundary",
        "cutoff",
        "no-cutoff",
        "reaction-field",
    ] = Field(
        _PME,
    )  # type: ignore[assignment]
    nonperiodic_potential: Literal[
        "Coulomb",
        "cutoff",
        "no-cutoff",
        "reaction-field",
    ] = Field(
        "Coulomb",
    )  # type: ignore[assignment]
    exception_potential: Literal["Coulomb"] = Field("Coulomb")

    _charges: dict[
        TopologyKey | LibraryChargeTopologyKey | VirtualSiteKey,
        _ElementaryChargeQuantity,
    ] = PrivateAttr(dict())
    _charges_cached: bool = PrivateAttr(default=False)

    @classmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        return [
            LibraryChargeHandler,
            ChargeIncrementModelHandler,
            ToolkitAM1BCCHandler,
            ElectrostaticsHandler,
        ]

    @classmethod
    def supported_parameters(cls):
        """Return a list of supported parameter attribute names."""

    @property
    def _charges_without_virtual_sites(
        self,
    ) -> dict[TopologyKey | LibraryChargeTopologyKey, _ElementaryChargeQuantity]:
        """Get the total partial charge on each atom, excluding virtual sites."""
        return self._get_charges(include_virtual_sites=False)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def charges(
        self,
    ) -> dict[TopologyKey | LibraryChargeTopologyKey | VirtualSiteKey, _ElementaryChargeQuantity]:
        """
        Get the total partial charge on each atom, including virtual sites, as a rich dict.

        For a simpler array-like representation, see `get_charge_array`.
        """
        if len(self._charges) == 0 or self._charges_cached is False:
            # TODO: Clearing this dict **should not be necessary** but in some cases in appears
            #       that this class attribute persists in some cases, possibly only on a single
            #       thread. Ideas to try include
            #           * Drop @computed_field ?
            #           * Don't handle caching ourselves and let Pydantic do it (i.e. have this
            #               function simply retrun ._get_charges(...)
            #
            #       Hopefully this isn't a major issue - caches for large systems should still
            #       only be build once
            #
            #       Some more context: https://github.com/openforcefield/openff-interchange/issues/842#issuecomment-2394211357
            self._charges.clear()

            self._charges.update(self._get_charges(include_virtual_sites=True))
            self._charges_cached = True

        return self._charges

    def get_charge_array(self, include_virtual_sites: bool = False) -> Quantity:
        """
        Return a one-dimensional array-like of atomic charges, ordered topologically.

        If virtual sites are present in the system, `NotImplementedError` is raised.
        """
        if include_virtual_sites:
            raise NotImplementedError("Not yet implemented with virtual sites")

        if VirtualSiteKey in {type(key) for key in self.key_map}:
            raise NotImplementedError(
                "Not yet implemented when virtual sites are present, even with `include_virtual_sites=False`.",
            )

        return Quantity.from_list([q for _, q in sorted(self.charges.items(), key=lambda x: x[0].atom_indices)])

    def _get_charges(
        self,
        include_virtual_sites: bool = True,
    ) -> dict[TopologyKey | LibraryChargeTopologyKey | VirtualSiteKey, _ElementaryChargeQuantity]:
        """Get the total partial charge on each atom or particle."""
        # Keyed by index for atoms and by VirtualSiteKey for virtual sites.
        # work in unitless (float, implicit elementary_charge) values until returning
        charges: dict[VirtualSiteKey | int, _ElementaryChargeQuantity] = dict()

        for topology_key, potential_key in self.key_map.items():
            potential = self.potentials[potential_key]

            for parameter_key, parameter_value in potential.parameters.items():
                if parameter_key == "charge_increments":
                    if type(topology_key) is not VirtualSiteKey:
                        raise RuntimeError(
                            '"charge_increments" (plural) understood to only be applied by'
                            "virtual sites, not by a `ChargeIncrementModelHandler`.",
                        )

                    # assumes virtual sites can only have charges determined in one step
                    charges[topology_key] = -1.0 * numpy.sum(parameter_value)

                    # Apply increments to "orientation" atoms
                    for i, increment in enumerate(parameter_value):
                        orientation_atom_index = topology_key.orientation_atom_indices[i]

                        charges[orientation_atom_index] = _add_charges(
                            charges.get(orientation_atom_index, _ZERO_CHARGE).m,
                            increment.m,
                        )

                elif parameter_key == "charge":
                    assert len(topology_key.atom_indices) == 1

                    atom_index = topology_key.atom_indices[0]

                    if potential_key.associated_handler in (
                        "LibraryCharges",
                        "ToolkitAM1BCCHandler",
                        "molecules_with_preset_charges",
                        "ExternalSource",
                    ):
                        charges[atom_index] = _add_charges(
                            charges.get(atom_index, _ZERO_CHARGE).m,
                            parameter_value.m,
                        )

                    elif potential_key.associated_handler in (  # type: ignore[operator]
                        "ChargeIncrementModelHandler"
                    ):
                        # the "charge" and "charge_increment" keys may not appear in that order, so
                        # we "add" the charge whether or not the increment was already applied.
                        # There should be a better way to do this.

                        charges[atom_index] = _add_charges(
                            charges.get(atom_index, _ZERO_CHARGE).m,
                            parameter_value.m,
                        )

                    else:
                        raise RuntimeError(
                            f"Unexepected associated handler {potential_key.associated_handler} found.",
                        )

                elif parameter_key == "charge_increment":
                    assert len(topology_key.atom_indices) == 1

                    atom_index = topology_key.atom_indices[0]

                    charges[atom_index] = _add_charges(
                        charges.get(atom_index, _ZERO_CHARGE).m,
                        parameter_value.m,
                    )

                    logger.info(
                        "Charge section ChargeIncrementModel, applying charge increment from atom "  # type: ignore[union-attr]
                        f"{topology_key.this_atom_index} to atoms {topology_key.other_atom_indices}",
                    )

                else:
                    raise NotImplementedError()

        returned_charges: dict[
            TopologyKey | LibraryChargeTopologyKey | VirtualSiteKey,
            Quantity,
        ] = dict()

        for index, charge in charges.items():
            if type(index) is int:
                # Note this uses the topology index, even if the particle index differs in an export,
                # i.e. if a molecule earlier in the topology has (collated) virtual sites as is
                # common with GROMACS files
                returned_charges[TopologyKey(atom_indices=(index,))] = charge
            elif type(index) is VirtualSiteKey:
                if include_virtual_sites:
                    returned_charges[index] = charge

        return {key: Quantity(val, "elementary_charge") for key, val in returned_charges.items()}

    @classmethod
    def parameter_handler_precedence(cls) -> list[str]:
        """
        Return the order in which parameter handlers take precedence when computing charges.
        """
        return ["LibraryCharges", "ChargeIncrementModel", "ToolkitAM1BCC"]

    @classmethod
    def create(
        cls,
        parameter_handler: ParameterHandler | list[ParameterHandler],
        topology: Topology,
        molecules_with_preset_charges=None,
        allow_nonintegral_charges: bool = False,
    ) -> Self:
        """
        Create a SMIRNOFFElectrostaticsCollection from toolkit data.

        """
        from packaging.version import Version

        if isinstance(parameter_handler, list):
            parameter_handlers = parameter_handler
        else:
            parameter_handlers = [parameter_handler]

        for handler in parameter_handlers:
            if isinstance(handler, ElectrostaticsHandler):
                if Version(str(handler.version)) < Version("0.4"):
                    raise SMIRNOFFVersionNotSupportedError(
                        "Electrostatics section must be up-converted to 0.4 or newer. Found version "
                        f"{handler.version}.",
                    )

        toolkit_handler_with_metadata = [p for p in parameter_handlers if type(p) is ElectrostaticsHandler][0]

        handler = cls(
            type=toolkit_handler_with_metadata.TAGNAME,
            scale_12=toolkit_handler_with_metadata.scale12,
            scale_13=toolkit_handler_with_metadata.scale13,
            scale_14=toolkit_handler_with_metadata.scale14,
            scale_15=toolkit_handler_with_metadata.scale15,
            cutoff=toolkit_handler_with_metadata.cutoff,
            periodic_potential=toolkit_handler_with_metadata.periodic_potential,
            nonperiodic_potential=toolkit_handler_with_metadata.nonperiodic_potential,
            exception_potential=toolkit_handler_with_metadata.exception_potential,
        )

        handler.store_matches(
            parameter_handlers,
            topology,
            molecules_with_preset_charges=molecules_with_preset_charges,
            allow_nonintegral_charges=allow_nonintegral_charges,
        )
        handler._charges = dict()

        return handler

    @classmethod
    @functools.lru_cache(None)
    def _compute_partial_charges(
        cls,
        molecule: Molecule,
        mapped_smiles: str,
        method: str,
    ) -> Quantity:
        """Call out to the toolkit's toolkit wrappers to generate partial charges."""
        molecule = copy.deepcopy(molecule)
        molecule.assign_partial_charges(method)

        return molecule.partial_charges

    @classmethod
    def _library_charge_to_potentials(
        cls,
        atom_indices: tuple[int, ...],
        parameter: LibraryChargeHandler.LibraryChargeType,
    ) -> tuple[
        dict[LibraryChargeTopologyKey, PotentialKey],
        dict[PotentialKey, Potential],
    ]:
        """
        Map a matched library charge parameter to a set of potentials.
        """
        matches = {}
        potentials = {}

        for i, (atom_index, charge) in enumerate(zip(atom_indices, parameter.charge)):
            topology_key = LibraryChargeTopologyKey(this_atom_index=atom_index)
            potential_key = PotentialKey(
                id=parameter.smirks,
                mult=i,
                associated_handler="LibraryCharges",
            )
            potential = Potential(parameters={"charge": charge})

            matches[topology_key] = potential_key
            potentials[potential_key] = potential

        return matches, potentials

    @classmethod
    def _charge_increment_to_potentials(
        cls,
        atom_indices: tuple[int, ...],
        parameter: ChargeIncrementModelHandler.ChargeIncrementType,
    ) -> tuple[
        dict[ChargeIncrementTopologyKey, PotentialKey],
        dict[PotentialKey, Potential],
    ]:
        """
        Map a matched charge increment parameter to a set of potentials.
        """
        matches = {}
        potentials = {}

        for i, atom_index in enumerate(atom_indices):
            other_atom_indices = tuple(val for val in atom_indices if val is not atom_index)
            topology_key = ChargeIncrementTopologyKey(
                this_atom_index=atom_index,
                other_atom_indices=other_atom_indices,
            )
            # TopologyKey(atom_indices=(atom_index,), mult=other_index)
            potential_key = PotentialKey(
                id=parameter.smirks,
                mult=i,
                associated_handler="ChargeIncrementModel",
            )

            # TODO: Handle the cases where n - 1 charge increments have been defined,
            #       maybe by implementing this in the TK?
            charge_increment = getattr(parameter, f"charge_increment{i + 1}")

            potential = Potential(parameters={"charge_increment": charge_increment})

            matches[topology_key] = potential_key
            potentials[potential_key] = potential

        return matches, potentials

    @classmethod
    def _find_slot_matches(
        cls,
        parameter_handler: LibraryChargeHandler | ChargeIncrementModelHandler,
        unique_molecule: Molecule,
    ) -> tuple[dict[TopologyKey, PotentialKey], dict[PotentialKey, Potential]]:
        """
        Construct a slot and potential map for a slot based parameter handler.
        """
        # Ideally this would be made redundant by OpenFF TK #971
        unique_parameter_matches = {
            tuple(sorted(key)): (key, val)
            for key, val in parameter_handler.find_matches(
                unique_molecule.to_topology(),
            ).items()
        }

        parameter_matches = {key: val for key, val in unique_parameter_matches.values()}
        if type(parameter_handler) is ChargeIncrementModelHandler:
            for atom_indices, val in parameter_matches.items():
                charge_increments = val.parameter_type.charge_increment

                if len(atom_indices) - len(charge_increments) == 0:
                    pass
                elif len(atom_indices) - len(charge_increments) == 1:
                    # If we've been provided with one less charge increment value than tagged atoms, assume the last
                    # tagged atom offsets the charge of the others to make the chargeincrement net-neutral
                    charge_increment_sum = Quantity(0.0, unit.elementary_charge)

                    for ci in charge_increments:
                        charge_increment_sum += ci
                    charge_increments.append(-charge_increment_sum)

                else:
                    from openff.toolkit.utils.exceptions import SMIRNOFFSpecError

                    raise SMIRNOFFSpecError(
                        f"Trying to apply chargeincrements {val.parameter_type} "
                        f"to tagged atoms {atom_indices}, but the number of chargeincrements "
                        "must be either the same as- or one less than the number of tagged atoms."
                        f"found {len(atom_indices)} number of tagged atoms and "
                        f"{len(val.parameter_type.charge_increment)} number of charge increments",
                    )

        matches, potentials = {}, {}

        for key, val in parameter_matches.items():
            parameter = val.parameter_type

            if isinstance(parameter_handler, LibraryChargeHandler):
                (
                    parameter_matches,
                    parameter_potentials,
                ) = cls._library_charge_to_potentials(key, parameter)

            elif isinstance(parameter_handler, ChargeIncrementModelHandler):
                (
                    parameter_matches,
                    parameter_potentials,
                ) = cls._charge_increment_to_potentials(key, parameter)

            else:
                raise NotImplementedError()

            for topology_key, potential_key in parameter_matches.items():
                # This may silently overwrite an identical key generated from a previous match, but that is
                # the toolkit behavior (see test_assign_charges_to_molecule_in_parts_using_multiple_library_charges).
                # If there is a need to track the topology keys that are ignored, this can be changed.
                matches[topology_key] = potential_key

            potentials.update(parameter_potentials)

        return matches, potentials

    @classmethod
    def _find_charge_model_matches(
        cls,
        parameter_handler: ToolkitAM1BCCHandler | ChargeIncrementModelHandler,
        unique_molecule: Molecule,
    ) -> tuple[
        str,
        dict[SingleAtomChargeTopologyKey, PotentialKey],
        dict[PotentialKey, Potential],
    ]:
        """Construct a slot and potential map for a charge model based parameter handler."""
        unique_molecule = copy.deepcopy(unique_molecule)
        reference_smiles = unique_molecule.to_smiles(
            isomeric=True,
            explicit_hydrogens=True,
            mapped=True,
        )

        handler_name = parameter_handler.__class__.__name__

        if handler_name == "ChargeIncrementModelHandler":
            partial_charge_method = parameter_handler.partial_charge_method
        elif handler_name == "ToolkitAM1BCCHandler":
            from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY

            # The implementation of _toolkit_registry_manager should result in this `GLOBAL_TOOLKIT_REGISTRY`
            # including only what it is passed, even if it's not what one would expect at import time
            if "OpenEye" in GLOBAL_TOOLKIT_REGISTRY.__repr__():
                partial_charge_method = "am1bccelf10"
            else:
                partial_charge_method = "am1bcc"
        else:
            raise InvalidParameterHandlerError(
                f"Encountered unknown handler of type {type(parameter_handler)} where only "
                "ToolkitAM1BCCHandler or ChargeIncrementModelHandler are expected.",
            )

        partial_charges = cls._compute_partial_charges(
            unique_molecule,
            unique_molecule.to_smiles(
                isomeric=True,
                explicit_hydrogens=True,
                mapped=True,
            ),
            method=partial_charge_method,
        )

        matches = {}
        potentials = {}

        for atom_index, partial_charge in enumerate(partial_charges):
            # These arguments make this object specific to this atom (by index) in this molecule ONLY
            # (assuming an isomeric, mapped, explicit hydrogen SMILES is unique, which seems true).
            potential_key = PotentialKey(
                id=reference_smiles,
                mult=atom_index,
                associated_handler=handler_name,
            )
            potentials[potential_key] = Potential(parameters={"charge": partial_charge})

            matches[
                SingleAtomChargeTopologyKey(
                    this_atom_index=atom_index,
                    extras={
                        "handler": handler_name,
                        "partial_charge_method": partial_charge_method,
                    },
                )
            ] = potential_key

        return partial_charge_method, matches, potentials

    @classmethod
    def _find_reference_matches(
        cls,
        parameter_handlers: dict[str, ElectrostaticsHandlerType],
        unique_molecule: Molecule,
    ) -> tuple[dict[TopologyKey, PotentialKey], dict[PotentialKey, Potential]]:
        """
        Construct a slot and potential map for a particular reference molecule and set of parameter handlers.
        """
        matches: dict[TopologyKey, PotentialKey] = dict()
        potentials: dict[PotentialKey, Potential] = dict()

        expected_matches = {i for i in range(unique_molecule.n_atoms)}

        for handler_type in cls.parameter_handler_precedence():
            if handler_type not in parameter_handlers:
                continue

            parameter_handler = parameter_handlers[handler_type]

            slot_matches, am1_matches = None, None
            slot_potentials: dict = {}
            am1_potentials: dict = {}

            if handler_type in ["LibraryCharges", "ChargeIncrementModel"]:
                slot_matches, slot_potentials = cls._find_slot_matches(
                    parameter_handler,
                    unique_molecule,
                )

            if handler_type in ["ToolkitAM1BCC", "ChargeIncrementModel"]:
                (
                    partial_charge_method,
                    am1_matches,
                    am1_potentials,
                ) = cls._find_charge_model_matches(
                    parameter_handler,
                    unique_molecule,
                )

            if slot_matches is None and am1_matches is None:
                raise NotImplementedError()

            elif slot_matches is not None and am1_matches is not None:
                am1_matches = {
                    ChargeModelTopologyKey(  # type: ignore[misc]
                        this_atom_index=topology_key.atom_indices[0],
                        partial_charge_method=partial_charge_method,
                    ): potential_key
                    for topology_key, potential_key in am1_matches.items()
                }

                matched_atom_indices = {index for key in slot_matches for index in key.atom_indices}
                matched_atom_indices.update(
                    {index for key in am1_matches for index in key.atom_indices},
                )

            elif slot_matches is not None:
                matched_atom_indices = {index for key in slot_matches for index in key.atom_indices}
            else:
                matched_atom_indices = {
                    index
                    for key in am1_matches  # type: ignore[union-attr]
                    for index in key.atom_indices
                }

            if matched_atom_indices != expected_matches:
                # Handle the case where a handler could not fully assign the charges
                # to the whole molecule.
                continue

            matches.update(slot_matches if slot_matches is not None else {})
            matches.update(am1_matches if am1_matches is not None else {})  # type: ignore[arg-type]

            potentials.update(slot_potentials)
            potentials.update(am1_potentials)

            break

        found_matches = {index for key in matches for index in key.atom_indices}

        if found_matches != expected_matches:
            raise RuntimeError(
                f"{unique_molecule.to_smiles(explicit_hydrogens=False)} could "
                "not be fully assigned charges. Charges were assigned to atoms "
                f"{found_matches} but the molecule contains {expected_matches}.",
            )

        return matches, potentials

    @classmethod
    def _assign_charges_from_molecules(
        cls,
        unique_molecule: Molecule,
        molecules_with_preset_charges=list[Molecule] | None,
    ) -> tuple[bool, dict, dict]:
        if molecules_with_preset_charges is None:
            return False, dict(), dict()

        for molecule_with_charges in molecules_with_preset_charges:
            if molecule_with_charges.is_isomorphic_with(unique_molecule):
                break
        else:
            return False, dict(), dict()

        _, atom_map = Molecule.are_isomorphic(
            molecule_with_charges,
            unique_molecule,
            return_atom_map=True,
        )

        from openff.interchange.models import SingleAtomChargeTopologyKey

        matches = dict()
        potentials = dict()
        mapped_smiles = unique_molecule.to_smiles(mapped=True, explicit_hydrogens=True)

        for index_in_molecule_with_charges, partial_charge in enumerate(
            molecule_with_charges.partial_charges,
        ):
            index_in_topology = atom_map[index_in_molecule_with_charges]
            topology_key = SingleAtomChargeTopologyKey(
                this_atom_index=index_in_topology,
                extras={"handler": "preset"},
            )
            potential_key = PotentialKey(
                id=mapped_smiles,
                mult=index_in_molecule_with_charges,  # Not sure this prevents clashes in some corner cases
                associated_handler="molecules_with_preset_charges",
                bond_order=None,
            )
            potential = Potential(parameters={"charge": partial_charge})
            matches[topology_key] = potential_key
            potentials[potential_key] = potential

        return True, matches, potentials

    def store_matches(
        self,
        parameter_handler: ElectrostaticsHandlerType | list[ElectrostaticsHandlerType],
        topology: Topology,
        molecules_with_preset_charges=None,
        allow_nonintegral_charges: bool = False,
    ) -> None:
        """
        Populate self.key_map with key-val pairs of slots and unique potential identifiers.
        """
        # Reshape the parameter handlers into a dictionary for easier referencing.
        parameter_handlers = {
            handler.TAGNAME: handler
            for handler in (parameter_handler if isinstance(parameter_handler, list) else [parameter_handler])
        }

        self.potentials = dict()
        self.key_map = dict()

        groups = topology.identical_molecule_groups

        for unique_molecule_index, group in groups.items():
            unique_molecule = topology.molecule(unique_molecule_index)

            flag, matches, potentials = self._assign_charges_from_molecules(
                unique_molecule,
                molecules_with_preset_charges,
            )
            # TODO: Here is where the toolkit calls self.check_charges_assigned(). Do we skip this
            #       entirely given that we are not accepting `molecules_with_preset_charges`?

            if not flag:
                # TODO: Rename this method to something like `_find_matches`
                matches, potentials = self._find_reference_matches(
                    parameter_handlers,
                    unique_molecule,
                )

            self.potentials.update(potentials)

            for unique_molecule_atom in unique_molecule.atoms:
                unique_molecule_atom_index = unique_molecule.atom_index(
                    unique_molecule_atom,
                )

                for duplicate_molecule_index, atom_map in group:
                    duplicate_molecule = topology.molecule(duplicate_molecule_index)
                    duplicate_molecule_atom_index = atom_map[unique_molecule_atom_index]
                    duplicate_molecule_atom = duplicate_molecule.atom(
                        duplicate_molecule_atom_index,
                    )
                    topology_atom_index = topology.atom_index(duplicate_molecule_atom)

                    # Copy the keys associated with the reference molecule to the duplicate molecule
                    for key in matches:
                        if key.this_atom_index == unique_molecule_atom_index:
                            new_key = key.__class__(**key.model_dump())
                            new_key.this_atom_index = topology_atom_index

                            # Have this new key (on a duplicate molecule) point to the same potential
                            # as the old key (on a unique/reference molecule)
                            if type(new_key) is LibraryChargeTopologyKey:
                                logger.info(
                                    "Charge section LibraryCharges applied to topology atom index "
                                    f"{topology_atom_index}",
                                )

                            elif type(new_key) is SingleAtomChargeTopologyKey:
                                if new_key.extras["handler"] == "ToolkitAM1BCCHandler":
                                    logger.info(
                                        "Charge section ToolkitAM1BCC, using charge method "
                                        f"{new_key.extras['partial_charge_method']}, "
                                        f"applied to topology atom index {topology_atom_index}",
                                    )

                                elif new_key.extras["handler"] == "preset":
                                    logger.info(
                                        f"Preset charges applied to atom index {topology_atom_index}",
                                    )

                                else:
                                    raise ValueError(f"Unhandled handler {new_key.extras['handler']}")

                            elif type(new_key) is ChargeModelTopologyKey:
                                logger.info(
                                    "Charge section ChargeIncrementModel, using charge method "
                                    f"{new_key.partial_charge_method}, "
                                    f"applied to topology atom index {new_key.this_atom_index}",
                                )

                            elif type(new_key) is ChargeIncrementTopologyKey:
                                # here is where the actual increments could be logged
                                pass

                            else:
                                raise ValueError(f"Unhandled key type {type(new_key)}")

                            self.key_map[new_key] = matches[key]

        topology_charges = [0.0] * topology.n_atoms

        for key, val in self.charges.items():
            topology_charges[key.atom_indices[0]] = val.m

        # TODO: Better data structures in Topology.identical_molecule_groups will make this
        #       cleaner and possibly more performant
        for molecule in topology.molecules:
            molecule_charges = [0.0] * molecule.n_atoms

            for atom in molecule.atoms:
                molecule_index = molecule.atom_index(atom)
                topology_index = topology.atom_index(atom)

                molecule_charges[molecule_index] = topology_charges[topology_index]

            charge_sum = sum(molecule_charges)
            formal_sum = molecule.total_charge.m

            if abs(charge_sum - formal_sum) > 0.01:
                if allow_nonintegral_charges:
                    # TODO: Is it worth communicating this as a warning, or would it simply be bloat?
                    pass
                else:
                    raise NonIntegralMoleculeChargeError(
                        f"Molecule {molecule.to_smiles(explicit_hydrogens=False)} has "
                        f"a net charge of {charge_sum} compared to a total formal charge of "
                        f"{formal_sum}.",
                    )

            molecule.partial_charges = Quantity(
                molecule_charges,
                unit.elementary_charge,
            )

    def store_potentials(
        self,
        parameter_handler: ElectrostaticsHandlerType | list[ElectrostaticsHandlerType],
    ) -> None:
        """
        Populate self.potentials with key-val pairs of [PotentialKey, Potential].

        """
        # This logic is handled by ``store_matches`` as we may need to create potentials
        # to store depending on the handler type.
