"""Models and utilities for processing SMIRNOFF data."""
import abc
import copy
import functools
import json
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff.parameters import (
    AngleHandler,
    BondHandler,
    ChargeIncrementModelHandler,
    ConstraintHandler,
    ElectrostaticsHandler,
    ImproperTorsionHandler,
    LibraryChargeHandler,
    ParameterHandler,
    ProperTorsionHandler,
    ToolkitAM1BCCHandler,
    UnassignedProperTorsionParameterException,
    UnassignedValenceParameterException,
    VirtualSiteHandler,
    vdWHandler,
)
from openff.units import unit
from openmm import unit as openmm_unit
from pydantic import Field
from typing_extensions import Literal

from openff.interchange.components.potentials import (
    Potential,
    PotentialHandler,
    WrappedPotential,
)
from openff.interchange.components.toolkit import _validated_list_to_array
from openff.interchange.constants import _PME
from openff.interchange.exceptions import (
    InvalidParameterHandlerError,
    MissingParametersError,
    NonIntegralMoleculeChargeException,
    SMIRNOFFParameterAttributeNotImplementedError,
    SMIRNOFFVersionNotSupportedError,
)
from openff.interchange.models import (
    ChargeIncrementTopologyKey,
    ChargeModelTopologyKey,
    LibraryChargeTopologyKey,
    PotentialKey,
    TopologyKey,
    VirtualSiteKey,
)
from openff.interchange.types import FloatQuantity, custom_quantity_encoder, json_loader

kcal_mol = openmm_unit.kilocalorie_per_mole
kcal_mol_angstroms = kcal_mol / openmm_unit.angstrom**2
kcal_mol_radians = kcal_mol / openmm_unit.radian**2

if TYPE_CHECKING:

    from openff.toolkit.topology import Topology
    from openff.units.unit import Quantity

    ElectrostaticsHandlerType = Union[
        ElectrostaticsHandler,
        ChargeIncrementModelHandler,
        LibraryChargeHandler,
        ToolkitAM1BCCHandler,
    ]


T = TypeVar("T", bound="SMIRNOFFPotentialHandler")
TP = TypeVar("TP", bound="PotentialHandler")


def _sanitize(o):
    # `BaseModel.json()` assumes that all keys and values in dicts are JSON-serializable, which is a problem
    # for the mapping dicts `slot_map` and `potentials`.
    if isinstance(o, dict):
        return {_sanitize(k): _sanitize(v) for k, v in o.items()}
    elif isinstance(o, (PotentialKey, TopologyKey)):
        return o.json()
    elif isinstance(o, unit.Quantity):
        return custom_quantity_encoder(o)
    return o


def handler_dumps(v, *, default):
    """Dump a SMIRNOFFPotentialHandler to JSON after converting to compatible types."""
    return json.dumps(_sanitize(v), default=default)


class SMIRNOFFPotentialHandler(PotentialHandler, abc.ABC):
    """Base class for handlers storing potentials produced by SMIRNOFF force fields."""

    class Config:
        """Default configuration options for SMIRNOFF potential handlers."""

        json_dumps = handler_dumps
        json_loads = json_loader
        validate_assignment = True
        arbitrary_types_allowed = True

    @classmethod
    @abc.abstractmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def supported_parameters(cls):
        """Return a list of parameter attributes supported by this handler."""
        raise NotImplementedError()

    #    @classmethod
    #    @abc.abstractmethod
    #    def valence_terms(cls, topology):
    #        """Return an interable of all of one type of valence term in this topology."""
    #        raise NotImplementedError()

    @classmethod
    def check_supported_parameters(cls, parameter_handler: ParameterHandler):
        """Verify that a parameter handler is in an allowed list of handlers."""
        for parameter in parameter_handler.parameters:
            for parameter_attribute in parameter._get_defined_parameter_attributes():
                if parameter_attribute == "parent_id":
                    continue
                if parameter_attribute not in cls.supported_parameters():
                    raise SMIRNOFFParameterAttributeNotImplementedError(
                        parameter_attribute,
                    )

    def store_matches(
        self,
        parameter_handler: ParameterHandler,
        topology: "Topology",
    ) -> None:
        """Populate self.slot_map with key-val pairs of [TopologyKey, PotentialKey]."""
        parameter_handler_name = getattr(parameter_handler, "_TAGNAME", None)
        if self.slot_map:
            # TODO: Should the slot_map always be reset, or should we be able to partially
            # update it? Also Note the duplicated code in the child classes
            self.slot_map = dict()
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            topology_key = TopologyKey(atom_indices=key)
            potential_key = PotentialKey(
                id=val.parameter_type.smirks, associated_handler=parameter_handler_name
            )
            self.slot_map[topology_key] = potential_key

        if self.__class__.__name__ in ["SMIRNOFFBondHandler", "SMIRNOFFAngleHandler"]:
            valence_terms = self.valence_terms(topology)  # type: ignore[attr-defined]

            _check_all_valence_terms_assigned(
                handler=parameter_handler,
                assigned_terms=matches,
                topology=topology,
                valence_terms=valence_terms,
                exception_cls=UnassignedValenceParameterException,
            )

    @classmethod
    def _from_toolkit(
        cls: Type[T],
        parameter_handler: TP,
        topology: "Topology",
    ) -> T:
        """
        Create a SMIRNOFFPotentialHandler from toolkit data.

        """
        if type(parameter_handler) not in cls.allowed_parameter_handlers():
            raise InvalidParameterHandlerError(type(parameter_handler))

        handler = cls()
        if hasattr(handler, "fractional_bondorder_method"):
            if getattr(parameter_handler, "fractional_bondorder_method", None):
                handler.fractional_bond_order_method = (  # type: ignore[attr-defined]
                    parameter_handler.fractional_bondorder_method  # type: ignore[attr-defined]
                )
                handler.fractional_bond_order_interpolation = (  # type: ignore[attr-defined]
                    parameter_handler.fractional_bondorder_interpolation  # type: ignore[attr-defined]
                )
        handler.store_matches(parameter_handler=parameter_handler, topology=topology)
        handler.store_potentials(parameter_handler=parameter_handler)

        return handler

    def __repr__(self) -> str:
        return (
            f"Handler '{self.type}' with expression '{self.expression}', {len(self.slot_map)} slots, "
            f"and {len(self.potentials)} potentials"
        )


class SMIRNOFFBondHandler(SMIRNOFFPotentialHandler):
    """Handler storing bond potentials as produced by a SMIRNOFF force field."""

    type: Literal["Bonds"] = "Bonds"
    expression: Literal["k/2*(r-length)**2"] = "k/2*(r-length)**2"
    fractional_bond_order_method: Literal["AM1-Wiberg", "None", "none"] = "AM1-Wiberg"
    fractional_bond_order_interpolation: Literal["linear"] = "linear"

    @classmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        return [BondHandler]

    @classmethod
    def supported_parameters(cls):
        """Return a list of supported parameter attribute names."""
        return ["smirks", "id", "k", "length", "k_bondorder", "length_bondorder"]

    @classmethod
    def valence_terms(cls, topology):
        """Return all bonds in this topology."""
        return [tuple(b.atoms) for b in topology.bonds]

    def store_matches(
        self,
        parameter_handler: ParameterHandler,
        topology: "Topology",
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots and unique potential identifiers.
        """
        parameter_handler_name = getattr(parameter_handler, "_TAGNAME", None)
        if self.slot_map:
            # TODO: Should the slot_map always be reset, or should we be able to partially
            # update it? Also Note the duplicated code in the child classes
            self.slot_map = dict()
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            param = val.parameter_type
            if param.k_bondorder or param.length_bondorder:
                bond = topology.get_bond_between(*key)
                fractional_bond_order = bond.fractional_bond_order
                if not fractional_bond_order:
                    assert self._get_uses_interpolation(parameter_handler)
                    raise RuntimeError(
                        "Bond orders should already be assigned at this point"
                    )
            else:
                fractional_bond_order = None
            topology_key = TopologyKey(
                atom_indices=key, bond_order=fractional_bond_order
            )
            potential_key = PotentialKey(
                id=val.parameter_type.smirks,
                associated_handler=parameter_handler_name,
                bond_order=fractional_bond_order,
            )
            self.slot_map[topology_key] = potential_key

        valence_terms = self.valence_terms(topology)

        _check_all_valence_terms_assigned(
            handler=parameter_handler,
            topology=topology,
            assigned_terms=matches,
            valence_terms=valence_terms,
            exception_cls=UnassignedValenceParameterException,
        )

    def store_potentials(self, parameter_handler: "BondHandler") -> None:
        """
        Populate self.potentials with key-val pairs of [TopologyKey, PotentialKey].

        """
        if self.potentials:
            self.potentials = dict()
        for topology_key, potential_key in self.slot_map.items():
            smirks = potential_key.id
            parameter = parameter_handler.get_parameter({"smirks": smirks})[0]
            if topology_key.bond_order:
                bond_order = topology_key.bond_order
                if parameter.k_bondorder:
                    data = parameter.k_bondorder
                else:
                    data = parameter.length_bondorder
                coeffs = _get_interpolation_coeffs(
                    fractional_bond_order=bond_order,
                    data=data,
                )
                pots = []
                map_keys = [*data.keys()]
                for map_key in map_keys:
                    pots.append(
                        Potential(
                            parameters={
                                "k": parameter.k_bondorder[map_key],
                                "length": parameter.length_bondorder[map_key],
                            },
                            map_key=map_key,
                        )
                    )
                potential = WrappedPotential(
                    {pot: coeff for pot, coeff in zip(pots, coeffs)}
                )
            else:
                potential = Potential(  # type: ignore[assignment]
                    parameters={
                        "k": parameter.k,
                        "length": parameter.length,
                    },
                )
            self.potentials[potential_key] = potential

    def _get_uses_interpolation(self, parameter_handler: "BondHandler") -> bool:
        if (
            any(
                getattr(p, "k_bondorder", None) is not None
                for p in parameter_handler.parameters
            )
        ) or (
            any(
                getattr(p, "length_bondorder", None) is not None
                for p in parameter_handler.parameters
            )
        ):
            return True
        else:
            return False

    @classmethod
    def _from_toolkit(
        cls: Type[T],
        parameter_handler: "BondHandler",
        topology: "Topology",
        partial_bond_orders_from_molecules=None,
    ) -> T:
        """
        Create a SMIRNOFFBondHandler from toolkit data.

        """
        # TODO: This method overrides SMIRNOFFPotentialHandler.from_toolkit in order to gobble up
        # a ConstraintHandler. This seems like a good solution for the interdependence, but is also
        # not a great practice. A better solution would involve not overriding the method with a
        # different function signature.
        if type(parameter_handler) not in cls.allowed_parameter_handlers():
            raise InvalidParameterHandlerError

        handler: T = cls(
            type="Bonds",
            expression="k/2*(r-length)**2",
            fractional_bond_order_method=parameter_handler.fractional_bondorder_method,
            fractional_bond_order_interpolation=parameter_handler.fractional_bondorder_interpolation,
        )

        if handler._get_uses_interpolation(parameter_handler):  # type: ignore[attr-defined]
            for molecule in topology.molecules:
                if _check_partial_bond_orders(
                    molecule, partial_bond_orders_from_molecules
                ):
                    continue
                # TODO: expose conformer generation and fractional bond order assigment
                # knobs to user via API
                molecule.generate_conformers(n_conformers=1)
                molecule.assign_fractional_bond_orders(
                    bond_order_model=handler.fractional_bond_order_method.lower(),  # type: ignore[attr-defined]
                )

        handler.store_matches(parameter_handler=parameter_handler, topology=topology)
        handler.store_potentials(parameter_handler=parameter_handler)

        return handler


class SMIRNOFFConstraintHandler(SMIRNOFFPotentialHandler):
    """Handler storing constraint potentials as produced by a SMIRNOFF force field."""

    type: Literal["Constraints"] = "Constraints"
    expression: Literal[""] = ""
    constraints: Dict[
        PotentialKey, bool
    ] = dict()  # should this be named potentials for consistency?

    @classmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        return [BondHandler, ConstraintHandler]

    @classmethod
    def supported_parameters(cls):
        """Return a list of supported parameter attribute names."""
        return ["smirks", "id", "k", "length", "distance"]

    @classmethod
    def _from_toolkit(  # type: ignore[override]
        cls: Type[T],
        parameter_handler: List,
        topology: "Topology",
    ) -> T:
        """
        Create a SMIRNOFFPotentialHandler from toolkit data.

        """
        if isinstance(parameter_handler, list):
            parameter_handlers = parameter_handler
        else:
            parameter_handlers = [parameter_handler]

        for parameter_handler in parameter_handlers:
            if type(parameter_handler) not in cls.allowed_parameter_handlers():
                raise InvalidParameterHandlerError(type(parameter_handler))

        handler = cls()
        handler.store_constraints(  # type: ignore[attr-defined]
            parameter_handlers=parameter_handlers, topology=topology
        )

        return handler

    def store_constraints(
        self,
        parameter_handlers: Any,
        topology: "Topology",
    ) -> None:
        """Store constraints."""
        if self.slot_map:
            self.slot_map = dict()

        constraint_handler = [
            p for p in parameter_handlers if type(p) == ConstraintHandler
        ][0]
        constraint_matches = constraint_handler.find_matches(topology)

        if any([type(p) == BondHandler for p in parameter_handlers]):
            bond_handler = [p for p in parameter_handlers if type(p) == BondHandler][0]
            bonds = SMIRNOFFBondHandler._from_toolkit(
                parameter_handler=bond_handler,
                topology=topology,
            )
        else:
            bond_handler = None
            bonds = None

        for key, match in constraint_matches.items():
            topology_key = TopologyKey(atom_indices=key)
            smirks = match.parameter_type.smirks
            distance = match.parameter_type.distance
            if distance is not None:
                # This constraint parameter is fully specified
                potential_key = PotentialKey(
                    id=smirks, associated_handler="Constraints"
                )
                self.slot_map[topology_key] = potential_key
                distance = match.parameter_type.distance
            else:
                # This constraint parameter depends on the BondHandler ...
                if bond_handler is None:
                    raise MissingParametersError(
                        f"Constraint with SMIRKS pattern {smirks} found with no distance "
                        "specified, and no corresponding bond parameters were found. The distance "
                        "of this constraint is not specified."
                    )
                # ... so use the same PotentialKey instance as the BondHandler to look up the distance
                potential_key = bonds.slot_map[topology_key]  # type: ignore[union-attr]
                self.slot_map[topology_key] = potential_key
                distance = bonds.potentials[potential_key].parameters["length"]  # type: ignore[union-attr]
            potential = Potential(
                parameters={
                    "distance": distance,
                }
            )
            self.constraints[potential_key] = potential  # type: ignore[assignment]


class SMIRNOFFAngleHandler(SMIRNOFFPotentialHandler):
    """Handler storing angle potentials as produced by a SMIRNOFF force field."""

    type: Literal["Angles"] = "Angles"
    expression: Literal["k/2*(theta-angle)**2"] = "k/2*(theta-angle)**2"

    @classmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        return [AngleHandler]

    @classmethod
    def supported_parameters(cls):
        """Return a list of supported parameter attributes."""
        return ["smirks", "id", "k", "angle"]

    @classmethod
    def valence_terms(cls, topology):
        """Return all angles in this topology."""
        return list(topology.angles)

    def store_potentials(self, parameter_handler: "AngleHandler") -> None:
        """
        Populate self.potentials with key-val pairs of [TopologyKey, PotentialKey].

        """
        for potential_key in self.slot_map.values():
            smirks = potential_key.id
            # ParameterHandler.get_parameter returns a list, although this
            # should only ever be length 1
            parameter = parameter_handler.get_parameter({"smirks": smirks})[0]
            potential = Potential(
                parameters={
                    "k": parameter.k,
                    "angle": parameter.angle,
                },
            )
            self.potentials[potential_key] = potential


class SMIRNOFFProperTorsionHandler(SMIRNOFFPotentialHandler):
    """Handler storing proper torsions potentials as produced by a SMIRNOFF force field."""

    type: Literal["ProperTorsions"] = "ProperTorsions"
    expression: Literal[
        "k*(1+cos(periodicity*theta-phase))"
    ] = "k*(1+cos(periodicity*theta-phase))"
    fractional_bond_order_method: Literal["AM1-Wiberg"] = "AM1-Wiberg"
    fractional_bond_order_interpolation: Literal["linear"] = "linear"

    @classmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        return [ProperTorsionHandler]

    @classmethod
    def supported_parameters(cls):
        """Return a list of supported parameter attribute names."""
        return ["smirks", "id", "k", "periodicity", "phase", "idivf", "k_bondorder"]

    def store_matches(
        self,
        parameter_handler: "ProperTorsionHandler",
        topology: "Topology",
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots and unique potential identifiers.

        """
        if self.slot_map:
            self.slot_map = dict()
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            param = val.parameter_type
            n_terms = len(val.parameter_type.phase)
            for n in range(n_terms):
                smirks = param.smirks
                if param.k_bondorder:
                    # The relevant bond order is that of the _central_ bond in the torsion
                    bond = topology.get_bond_between(key[1], key[2])
                    fractional_bond_order = bond.fractional_bond_order
                    if not fractional_bond_order:
                        raise RuntimeError(
                            "Bond orders should already be assigned at this point"
                        )
                else:
                    fractional_bond_order = None
                topology_key = TopologyKey(
                    atom_indices=key, mult=n, bond_order=fractional_bond_order
                )
                potential_key = PotentialKey(
                    id=smirks,
                    mult=n,
                    associated_handler="ProperTorsions",
                    bond_order=fractional_bond_order,
                )
                self.slot_map[topology_key] = potential_key

        _check_all_valence_terms_assigned(
            handler=parameter_handler,
            topology=topology,
            assigned_terms=matches,
            valence_terms=list(topology.propers),
            exception_cls=UnassignedProperTorsionParameterException,
        )

    def store_potentials(self, parameter_handler: "ProperTorsionHandler") -> None:
        """
        Populate self.potentials with key-val pairs of [TopologyKey, PotentialKey].

        """
        for topology_key, potential_key in self.slot_map.items():
            smirks = potential_key.id
            n = potential_key.mult
            parameter = parameter_handler.get_parameter({"smirks": smirks})[0]
            # n_terms = len(parameter.k)
            if topology_key.bond_order:
                bond_order = topology_key.bond_order
                data = parameter.k_bondorder[n]
                coeffs = _get_interpolation_coeffs(
                    fractional_bond_order=bond_order,
                    data=data,
                )
                pots = []
                map_keys = [*data.keys()]
                for map_key in map_keys:
                    parameters = {
                        "k": parameter.k_bondorder[n][map_key],
                        "periodicity": parameter.periodicity[n] * unit.dimensionless,
                        "phase": parameter.phase[n],
                        "idivf": parameter.idivf[n] * unit.dimensionless,
                    }
                    pots.append(
                        Potential(
                            parameters=parameters,
                            map_key=map_key,
                        )
                    )
                potential = WrappedPotential(
                    {pot: coeff for pot, coeff in zip(pots, coeffs)}
                )
            else:
                parameters = {
                    "k": parameter.k[n],
                    "periodicity": parameter.periodicity[n] * unit.dimensionless,
                    "phase": parameter.phase[n],
                    "idivf": parameter.idivf[n] * unit.dimensionless,
                }
                potential = Potential(parameters=parameters)  # type: ignore[assignment]
            self.potentials[potential_key] = potential

    @classmethod
    def _from_toolkit(
        cls: Type[T],
        parameter_handler: "ProperTorsionHandler",
        topology: "Topology",
        partial_bond_orders_from_molecules=None,
    ) -> T:
        """
        Create a SMIRNOFFProperTorsionHandler from toolkit data.

        """
        handler: T = cls(
            type="ProperTorsions",
            expression="k*(1+cos(periodicity*theta-phase))",
            fractional_bond_order_method=parameter_handler.fractional_bondorder_method,
            fractional_bond_order_interpolation=parameter_handler.fractional_bondorder_interpolation,
        )

        if any(
            getattr(p, "k_bondorder", None) is not None
            for p in parameter_handler.parameters
        ):
            for ref_mol in topology.reference_molecules:
                if _check_partial_bond_orders(
                    ref_mol, partial_bond_orders_from_molecules
                ):
                    continue
                # TODO: expose conformer generation and fractional bond order assigment knobs via API?
                ref_mol.generate_conformers(n_conformers=1)
                ref_mol.assign_fractional_bond_orders(
                    bond_order_model=handler.fractional_bond_order_method.lower(),  # type: ignore[attr-defined]
                )

        handler.store_matches(parameter_handler=parameter_handler, topology=topology)
        handler.store_potentials(parameter_handler=parameter_handler)

        return handler


class SMIRNOFFImproperTorsionHandler(SMIRNOFFPotentialHandler):
    """Handler storing improper torsions potentials as produced by a SMIRNOFF force field."""

    type: Literal["ImproperTorsions"] = "ImproperTorsions"
    expression: Literal[
        "k*(1+cos(periodicity*theta-phase))"
    ] = "k*(1+cos(periodicity*theta-phase))"
    # TODO: Consider whether or not default_idivf should be stored here

    @classmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        return [ImproperTorsionHandler]

    @classmethod
    def supported_parameters(cls):
        """Return a list of supported parameter attribute names."""
        return ["smirks", "id", "k", "periodicity", "phase", "idivf"]

    def store_matches(
        self,
        parameter_handler: "ImproperTorsionHandler",
        topology: "Topology",
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots and unique potential identifiers.

        """
        if self.slot_map:
            self.slot_map = dict()
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            parameter_handler._assert_correct_connectivity(
                val,
                [
                    (0, 1),
                    (1, 2),
                    (1, 3),
                ],
            )
            n_terms = len(val.parameter_type.k)
            for n in range(n_terms):

                smirks = val.parameter_type.smirks
                non_central_indices = [key[0], key[2], key[3]]

                for permuted_key in [
                    (
                        non_central_indices[i],
                        non_central_indices[j],
                        non_central_indices[k],
                    )
                    for (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
                ]:

                    topology_key = TopologyKey(
                        atom_indices=(key[1], *permuted_key), mult=n
                    )
                    potential_key = PotentialKey(
                        id=smirks, mult=n, associated_handler="ImproperTorsions"
                    )
                    self.slot_map[topology_key] = potential_key

    def store_potentials(self, parameter_handler: "ImproperTorsionHandler") -> None:
        """
        Populate self.potentials with key-val pairs of [TopologyKey, PotentialKey].

        """
        _default_idivf = parameter_handler.default_idivf

        for potential_key in self.slot_map.values():
            smirks = potential_key.id
            n = potential_key.mult
            parameter = parameter_handler.get_parameter({"smirks": smirks})[0]
            if parameter.idivf is None:
                idivf = None
            else:
                # Assumed to be list here
                idivf = parameter.idivf[n]
                if idivf is not None:
                    idivf = idivf * unit.dimensionless

            if idivf is None:
                if _default_idivf == "auto":
                    idivf = 3.0 * unit.dimensionless
                else:
                    # Assumed to be a numerical value
                    idivf = _default_idivf * unit.dimensionless

            parameters = {
                "k": parameter.k[n],
                "periodicity": parameter.periodicity[n] * unit.dimensionless,
                "phase": parameter.phase[n],
                "idivf": idivf,
            }
            potential = Potential(parameters=parameters)
            self.potentials[potential_key] = potential


class _SMIRNOFFNonbondedHandler(SMIRNOFFPotentialHandler, abc.ABC):
    """Base class for handlers storing non-bonded potentials produced by SMIRNOFF force fields."""

    type: str = "nonbonded"

    cutoff: FloatQuantity["angstrom"] = Field(
        9.0 * unit.angstrom,
        description="The distance at which pairwise interactions are truncated",
    )

    scale_13: float = Field(
        0.0, description="The scaling factor applied to 1-3 interactions"
    )
    scale_14: float = Field(
        0.5, description="The scaling factor applied to 1-4 interactions"
    )
    scale_15: float = Field(
        1.0, description="The scaling factor applied to 1-5 interactions"
    )


class SMIRNOFFvdWHandler(_SMIRNOFFNonbondedHandler):
    """Handler storing vdW potentials as produced by a SMIRNOFF force field."""

    type: Literal["vdW"] = "vdW"

    expression: Literal[
        "4*epsilon*((sigma/r)**12-(sigma/r)**6)"
    ] = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"

    method: Literal["cutoff", "pme", "no-cutoff"] = Field("cutoff")

    mixing_rule: Literal["lorentz-berthelot", "geometric"] = Field(
        "lorentz-berthelot",
        description="The mixing rule (combination rule) used in computing pairwise vdW interactions",
    )

    switch_width: FloatQuantity["angstrom"] = Field(
        1.0 * unit.angstrom,
        description="The width over which the switching function is applied",
    )

    @classmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        return [vdWHandler]

    @classmethod
    def supported_parameters(cls):
        """Return a list of supported parameter attributes."""
        return ["smirks", "id", "sigma", "epsilon", "rmin_half"]

    def store_potentials(self, parameter_handler: vdWHandler) -> None:
        """
        Populate self.potentials with key-val pairs of [TopologyKey, PotentialKey].

        """
        self.method = parameter_handler.method.lower()
        self.cutoff = parameter_handler.cutoff

        for potential_key in self.slot_map.values():
            smirks = potential_key.id
            parameter = parameter_handler.get_parameter({"smirks": smirks})[0]
            try:
                potential = Potential(
                    parameters={
                        "sigma": parameter.sigma,
                        "epsilon": parameter.epsilon,
                    },
                )
            except AttributeError:
                # Handle rmin_half pending https://github.com/openforcefield/openff-toolkit/pull/750
                potential = Potential(
                    parameters={
                        "sigma": parameter.sigma,
                        "epsilon": parameter.epsilon,
                    },
                )
            self.potentials[potential_key] = potential

    @classmethod
    def _from_toolkit(
        cls: Type[T],
        parameter_handler: "vdWHandler",
        topology: "Topology",
    ) -> T:
        """
        Create a SMIRNOFFvdWHandler from toolkit data.

        """
        if isinstance(parameter_handler, list):
            parameter_handlers = parameter_handler
        else:
            parameter_handlers = [parameter_handler]

        for parameter_handler in parameter_handlers:
            if type(parameter_handler) not in cls.allowed_parameter_handlers():
                raise InvalidParameterHandlerError(
                    f"Found parameter handler type {type(parameter_handler)}, which is not "
                    f"supported by potential type {type(cls)}"
                )

        handler = cls(
            scale_13=parameter_handler.scale13,
            scale_14=parameter_handler.scale14,
            scale_15=parameter_handler.scale15,
            cutoff=parameter_handler.cutoff,
            mixing_rule=parameter_handler.combining_rules.lower(),
            method=parameter_handler.method.lower(),
            switch_width=parameter_handler.switch_width,
        )
        handler.store_matches(parameter_handler=parameter_handler, topology=topology)
        handler.store_potentials(parameter_handler=parameter_handler)

        return handler

    @classmethod
    def parameter_handler_precedence(cls) -> List[str]:
        """
        Return the order in which parameter handlers take precedence when computing charges.
        """
        return ["vdw", "VirtualSites"]

    def _from_toolkit_virtual_sites(
        self,
        parameter_handler: "VirtualSiteHandler",
        topology: "Topology",
    ):
        # TODO: Merge this logic into _from_toolkit

        if not all(
            isinstance(
                p,
                (VirtualSiteHandler.VirtualSiteType,),
            )
            for p in parameter_handler.parameters
        ):
            raise NotImplementedError("Found unsupported virtual site types")

        matches = parameter_handler.find_matches(topology)
        for atoms, parameter_match in matches.items():
            virtual_site_type = parameter_match[0].parameter_type
            top_key = VirtualSiteKey(
                atom_indices=atoms,
                type=virtual_site_type.type,
                name=virtual_site_type.name,
                match=virtual_site_type.match,
            )
            pot_key = PotentialKey(
                id=virtual_site_type.smirks, associated_handler=virtual_site_type.type
            )
            pot = Potential(
                parameters={
                    "sigma": virtual_site_type.sigma,
                    "epsilon": virtual_site_type.epsilon,
                    # "distance": virtual_site_type.distance,
                }
            )
            # if virtual_site_type.type in {"MonovalentLonePair", "DivalentLonePair"}:
            #     pot.parameters.update(
            #         {
            #             "outOfPlaneAngle": virtual_site_type.outOfPlaneAngle,
            #         }
            #     )
            # if virtual_site_type.type in {"MonovalentLonePair"}:
            #     pot.parameters.update(
            #         {
            #             "inPlaneAngle": virtual_site_type.inPlaneAngle,
            #         }
            #     )

            self.slot_map.update({top_key: pot_key})  # type: ignore[dict-item]
            self.potentials.update({pot_key: pot})


class SMIRNOFFElectrostaticsHandler(_SMIRNOFFNonbondedHandler):
    """
    A handler which stores any electrostatic parameters applied to a topology.

    This handler is responsible for grouping together

    * global settings for the electrostatic interactions such as the cutoff distance
      and the intramolecular scale factors.
    * partial charges which have been assigned by a ``ToolkitAM1BCC``,
      ``LibraryCharges``, or a ``ChargeIncrementModel`` parameter
      handler.
    * charge corrections applied by a ``SMIRNOFFChargeIncrementHandler``.

    rather than having each in their own handler.
    """

    type: Literal["Electrostatics"] = "Electrostatics"
    expression: Literal["coul"] = "coul"

    periodic_potential: Literal[
        "Ewald3D-ConductingBoundary", "cutoff", "no-cutoff"
    ] = Field(_PME)
    nonperiodic_potential: Literal["Coulomb", "cutoff", "no-cutoff"] = Field("Coulomb")
    exception_potential: Literal["Coulomb"] = Field("Coulomb")

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
        pass

    @property
    def charges(self) -> Dict[Union[TopologyKey, VirtualSiteKey], "Quantity"]:
        """Get the total partial charge on each atom, excluding virtual sites."""
        return self.get_charges(include_virtual_sites=False)

    @property
    def charges_with_virtual_sites(
        self,
    ) -> Dict[Union[VirtualSiteKey, TopologyKey], "Quantity"]:
        """Get the total partial charge on each atom, including virtual sites."""
        return self.get_charges(include_virtual_sites=True)

    def get_charges(
        self, include_virtual_sites=False
    ) -> Dict[Union[VirtualSiteKey, TopologyKey], "Quantity"]:
        """Get the total partial charge on each atom or particle."""
        charges: DefaultDict[Union[TopologyKey, VirtualSiteKey], float] = defaultdict(
            lambda: 0.0
        )

        for topology_key, potential_key in self.slot_map.items():

            potential = self.potentials[potential_key]

            for parameter_key, parameter_value in potential.parameters.items():

                if parameter_key == "charge_increments":

                    if type(topology_key) != VirtualSiteKey:
                        raise RuntimeError

                    total_charge = np.sum(parameter_value)
                    # assumes virtual sites can only have charges determined in one step
                    # here, topology_key is actually a VirtualSiteKey
                    charges[topology_key] = -1.0 * total_charge

                    # Apply increments to "orientation" atoms
                    for i, increment in enumerate(parameter_value):
                        orientation_atom_key = TopologyKey(
                            atom_indices=(topology_key.orientation_atom_indices[i],)
                        )
                        charges[orientation_atom_key] += increment

                elif parameter_key in ["charge", "charge_increment"]:
                    charge = parameter_value
                    assert len(topology_key.atom_indices) == 1
                    charges[topology_key.atom_indices[0]] += charge  # type: ignore
                else:
                    raise NotImplementedError()

        returned_charges: Dict[Union[VirtualSiteKey, TopologyKey], "Quantity"] = dict()

        for index, charge in charges.items():
            if isinstance(index, int):
                returned_charges[TopologyKey(atom_indices=(index,))] = charge
            else:
                if include_virtual_sites:
                    returned_charges[index] = charge

        return returned_charges

    @classmethod
    def parameter_handler_precedence(cls) -> List[str]:
        """
        Return the order in which parameter handlers take precedence when computing charges.
        """
        return ["LibraryCharges", "ChargeIncrementModel", "ToolkitAM1BCC"]

    @classmethod
    def _from_toolkit(
        cls: Type[T],
        parameter_handler: Any,
        topology: "Topology",
        charge_from_molecules=None,
        allow_nonintegral_charges: bool = False,
    ) -> T:
        """
        Create a SMIRNOFFElectrostaticsHandler from toolkit data.

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
                        f"{handler.version}."
                    )

        toolkit_handler_with_metadata = [
            p for p in parameter_handlers if type(p) == ElectrostaticsHandler
        ][0]

        handler = cls(
            type=toolkit_handler_with_metadata._TAGNAME,
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
            charge_from_molecules=charge_from_molecules,
            allow_nonintegral_charges=allow_nonintegral_charges,
        )

        return handler

    def _from_toolkit_virtual_sites(
        self,
        parameter_handler: "VirtualSiteHandler",
        topology: "Topology",
    ):
        # TODO: Merge this logic into _from_toolkit

        if not all(
            isinstance(
                p,
                (
                    VirtualSiteHandler.VirtualSiteBondChargeType,
                    VirtualSiteHandler.VirtualSiteMonovalentLonePairType,
                    VirtualSiteHandler.VirtualSiteDivalentLonePairType,
                    VirtualSiteHandler.VirtualSiteTrivalentLonePairType,
                ),
            )
            for p in parameter_handler.parameters
        ):
            raise NotImplementedError("Found unsupported virtual site types")

        matches = parameter_handler.find_matches(topology)
        for atom_indices, parameter_match in matches.items():
            virtual_site_type = parameter_match[0].parameter_type

            virtual_site_key = VirtualSiteKey(
                atom_indices=atom_indices,
                type=virtual_site_type.type,
                name=virtual_site_type.name,
                match=virtual_site_type.match,
            )

            virtual_site_potential_key = PotentialKey(
                id=virtual_site_type.smirks,
                associated_handler="VirtualSiteHandler",
            )

            virtual_site_potential = Potential(
                parameters={
                    "charge_increments": _validated_list_to_array(
                        virtual_site_type.charge_increment
                    ),
                }
            )

            self.slot_map.update({virtual_site_key: virtual_site_potential_key})  # type: ignore[dict-item]
            self.potentials.update({virtual_site_potential_key: virtual_site_potential})

            for i, atom_index in enumerate(atom_indices):  # noqa
                topology_key = TopologyKey(atom_indices=(atom_index,), mult=i)

                # TODO: Better way of dedupliciating this case (charge increments from multiple different
                #       virtual sites are applied to the same atom)
                while topology_key in self.slot_map:
                    topology_key.mult += 1000  # type: ignore[operator]

                potential_key = PotentialKey(
                    id=virtual_site_type.smirks,
                    mult=i,
                    associated_handler="VirtualSiteHandler",
                )

                charge_increment = getattr(
                    virtual_site_type, f"charge_increment{i + 1}"
                )

                potential = Potential(parameters={"charge_increment": charge_increment})

                self.slot_map[topology_key] = potential_key
                self.potentials[potential_key] = potential

    @classmethod
    @functools.lru_cache(None)
    def _compute_partial_charges(cls, molecule: Molecule, method: str) -> "Quantity":
        """Call out to the toolkit's toolkit wrappers to generate partial charges."""
        molecule = copy.deepcopy(molecule)
        molecule.assign_partial_charges(method)

        return molecule.partial_charges

    @classmethod
    def _library_charge_to_potentials(
        cls,
        atom_indices: Tuple[int, ...],
        parameter: LibraryChargeHandler.LibraryChargeType,
    ) -> Tuple[Dict[TopologyKey, PotentialKey], Dict[PotentialKey, Potential]]:
        """
        Map a matched library charge parameter to a set of potentials.
        """
        matches = {}
        potentials = {}

        for i, (atom_index, charge) in enumerate(zip(atom_indices, parameter.charge)):
            topology_key = LibraryChargeTopologyKey(this_atom_index=atom_index)
            potential_key = PotentialKey(
                id=parameter.smirks, mult=i, associated_handler="LibraryCharges"
            )
            potential = Potential(parameters={"charge": charge})

            matches[topology_key] = potential_key
            potentials[potential_key] = potential

        return matches, potentials  # type: ignore[return-value]

    @classmethod
    def _charge_increment_to_potentials(
        cls,
        atom_indices: Tuple[int, ...],
        parameter: ChargeIncrementModelHandler.ChargeIncrementType,
    ) -> Tuple[Dict[TopologyKey, PotentialKey], Dict[PotentialKey, Potential]]:
        """
        Map a matched charge increment parameter to a set of potentials.
        """
        matches = {}
        potentials = {}

        for i, atom_index in enumerate(atom_indices):
            other_atom_indices = tuple(
                val for val in atom_indices if val is not atom_index
            )
            topology_key = ChargeIncrementTopologyKey(
                this_atom_index=atom_index,
                other_atom_indices=other_atom_indices,
            )
            # TopologyKey(atom_indices=(atom_index,), mult=other_index)
            potential_key = PotentialKey(
                id=parameter.smirks, mult=i, associated_handler="ChargeIncrementModel"
            )

            # TODO: Handle the cases where n - 1 charge increments have been defined,
            #       maybe by implementing this in the TK?
            charge_increment = getattr(parameter, f"charge_increment{i + 1}")

            potential = Potential(parameters={"charge_increment": charge_increment})

            matches[topology_key] = potential_key
            potentials[potential_key] = potential

        return matches, potentials  # type: ignore[return-value]

    @classmethod
    def _find_slot_matches(
        cls,
        parameter_handler: Union["LibraryChargeHandler", "ChargeIncrementModelHandler"],
        unique_molecule: Molecule,
    ) -> Tuple[Dict[TopologyKey, PotentialKey], Dict[PotentialKey, Potential]]:
        """
        Construct a slot and potential map for a slot based parameter handler.
        """
        # Ideally this would be made redundant by OpenFF TK #971
        unique_parameter_matches = {
            tuple(sorted(key)): (key, val)
            for key, val in parameter_handler.find_matches(
                unique_molecule.to_topology()
            ).items()
        }

        parameter_matches = {key: val for key, val in unique_parameter_matches.values()}
        if type(parameter_handler) == ChargeIncrementModelHandler:
            for atom_indices, val in parameter_matches.items():
                charge_increments = val.parameter_type.charge_increment

                if len(atom_indices) - len(charge_increments) == 0:
                    pass
                elif len(atom_indices) - len(charge_increments) == 1:
                    # If we've been provided with one less charge increment value than tagged atoms, assume the last
                    # tagged atom offsets the charge of the others to make the chargeincrement net-neutral
                    charge_increment_sum = unit.Quantity(0.0, unit.elementary_charge)

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
                        f"{len(val.parameter_type.charge_increment)} number of charge increments"
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
        parameter_handler: Union["ToolkitAM1BCCHandler", ChargeIncrementModelHandler],
        unique_molecule: Molecule,
    ) -> Tuple[str, Dict[TopologyKey, PotentialKey], Dict[PotentialKey, Potential]]:
        """Construct a slot and potential map for a charge model based parameter handler."""
        from openff.interchange.models import SingleAtomChargeTopologyKey

        unique_molecule = copy.deepcopy(unique_molecule)
        reference_smiles = unique_molecule.to_smiles(
            isomeric=True, explicit_hydrogens=True, mapped=True
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
                "ToolkitAM1BCCHandler or ChargeIncrementModelHandler are expected."
            )

        partial_charges = cls._compute_partial_charges(
            unique_molecule, method=partial_charge_method
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
                SingleAtomChargeTopologyKey(this_atom_index=atom_index)
            ] = potential_key

        return partial_charge_method, matches, potentials  # type: ignore[return-value]

    @classmethod
    def _find_reference_matches(
        cls,
        parameter_handlers: Dict[str, "ElectrostaticsHandlerType"],
        unique_molecule: Molecule,
    ) -> Tuple[Dict[TopologyKey, PotentialKey], Dict[PotentialKey, Potential]]:
        """
        Construct a slot and potential map for a particular reference molecule and set of parameter handlers.
        """
        matches = {}
        potentials = {}

        expected_matches = {i for i in range(unique_molecule.n_atoms)}

        for handler_type in cls.parameter_handler_precedence():

            if handler_type not in parameter_handlers:
                continue

            parameter_handler = parameter_handlers[handler_type]

            slot_matches, am1_matches = None, None
            slot_potentials: Dict = {}
            am1_potentials: Dict = {}

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

                matched_atom_indices = {
                    index for key in slot_matches for index in key.atom_indices
                }
                matched_atom_indices.update(
                    {index for key in am1_matches for index in key.atom_indices}
                )

            elif slot_matches is not None:
                matched_atom_indices = {
                    index for key in slot_matches for index in key.atom_indices
                }
            else:
                matched_atom_indices = {
                    index for key in am1_matches for index in key.atom_indices  # type: ignore[union-attr]
                }

            if matched_atom_indices != expected_matches:
                # Handle the case where a handler could not fully assign the charges
                # to the whole molecule.
                continue

            matches.update(slot_matches if slot_matches is not None else {})
            matches.update(am1_matches if am1_matches is not None else {})

            potentials.update(slot_potentials)
            potentials.update(am1_potentials)

            break

        found_matches = {index for key in matches for index in key.atom_indices}

        if found_matches != expected_matches:

            raise RuntimeError(
                f"{unique_molecule.to_smiles(explicit_hydrogens=False)} could "
                "not be fully assigned charges. Charges were assigned to atoms "
                f"{found_matches} but the molecule contains {expected_matches}."
            )

        return matches, potentials

    @classmethod
    def _assign_charges_from_molecules(
        cls,
        topology: "Topology",
        unique_molecule: Molecule,
        charge_from_molecules=Optional[List[Molecule]],
    ) -> Tuple[bool, Dict, Dict]:
        if charge_from_molecules is None:
            return False, dict(), dict()

        for molecule_with_charges in charge_from_molecules:
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
            molecule_with_charges.partial_charges
        ):
            index_in_topology = atom_map[index_in_molecule_with_charges]
            topology_key = SingleAtomChargeTopologyKey(
                this_atom_index=index_in_topology
            )
            potential_key = PotentialKey(
                id=mapped_smiles,
                mult=index_in_molecule_with_charges,  # Not sure this prevents clashes in some corner cases
                associated_handler="charge_from_molecules",
                bond_order=None,
            )
            potential = Potential(parameters={"charge": partial_charge})
            matches[topology_key] = potential_key
            potentials[potential_key] = potential

        return True, matches, potentials

    def store_matches(
        self,
        parameter_handler: Union[
            "ElectrostaticsHandlerType", List["ElectrostaticsHandlerType"]
        ],
        topology: "Topology",
        charge_from_molecules=None,
        allow_nonintegral_charges: bool = False,
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots and unique potential identifiers.
        """
        # Reshape the parameter handlers into a dictionary for easier referencing.
        parameter_handlers = {
            handler._TAGNAME: handler
            for handler in (
                parameter_handler
                if isinstance(parameter_handler, list)
                else [parameter_handler]
            )
        }

        self.potentials = dict()
        self.slot_map = dict()

        groups = topology.identical_molecule_groups

        for unique_molecule_index, group in groups.items():

            unique_molecule = topology.molecule(unique_molecule_index)

            flag, matches, potentials = self._assign_charges_from_molecules(
                topology,
                unique_molecule,
                charge_from_molecules,
            )
            # TODO: Here is where the toolkit calls self.check_charges_assigned(). Do we skip this
            #       entirely given that we are not accepting `charge_from_molecules`?

            if not flag:
                # TODO: Rename this method to something like `_find_matches`
                matches, potentials = self._find_reference_matches(
                    parameter_handlers,
                    unique_molecule,
                )

            self.potentials.update(potentials)

            for unique_molecule_atom in unique_molecule.atoms:
                unique_molecule_atom_index = unique_molecule.atom_index(
                    unique_molecule_atom
                )

                for duplicate_molecule_index, atom_map in group:
                    duplicate_molecule = topology.molecule(duplicate_molecule_index)
                    duplicate_molecule_atom_index = atom_map[unique_molecule_atom_index]
                    duplicate_molecule_atom = duplicate_molecule.atom(
                        duplicate_molecule_atom_index
                    )
                    topology_atom_index = topology.atom_index(duplicate_molecule_atom)

                    # Copy the keys associated with the reference molecule to the duplicate molecule
                    for key in matches:
                        if key.this_atom_index == unique_molecule_atom_index:
                            new_key = key.__class__(**key.dict())
                            new_key.this_atom_index = topology_atom_index

                            # Have this new key (on a duplicate molecule) point to the same potential
                            # as the old key (on a unique/reference molecule)
                            self.slot_map[new_key] = matches[key]

                    # for key in _slow_key_lookup_by_atom_index(
                    #     matches,
                    #     topology_atom_index,
                    # ):
                    #     self.slot_map[key] = matches[key]

        topology_charges = [0.0] * topology.n_atoms
        for key, val in self.get_charges().items():
            topology_charges[key.atom_indices[0]] = val.m
        # charges: List[float] = [v.m for v in self.get_charges().values()]

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
                    raise NonIntegralMoleculeChargeException(
                        f"Molecule {molecule.to_smiles(explicit_hydrogens=False)} has "
                        f"a net charge of {charge_sum}"
                    )

            molecule.partial_charges = unit.Quantity(
                molecule_charges, unit.elementary_charge
            )

    def store_potentials(
        self,
        parameter_handler: Union[
            "ElectrostaticsHandlerType", List["ElectrostaticsHandlerType"]
        ],
    ) -> None:
        """
        Populate self.potentials with key-val pairs of [TopologyKey, PotentialKey].

        """
        # This logic is handled by ``store_matches`` as we may need to create potentials
        # to store depending on the handler type.
        pass


class SMIRNOFFVirtualSiteHandler(SMIRNOFFPotentialHandler):
    """
    A handler which stores the information necessary to construct virtual sites (virtual particles).
    """

    slot_map: Dict[VirtualSiteKey, PotentialKey] = Field(
        dict(),
        description="A mapping between VirtualSiteKey objects and PotentialKey objects.",
    )  # type: ignore[assignment]

    type: Literal["VirtualSites"] = "VirtualSites"
    expression: Literal[""] = ""
    virtual_site_key_topology_index_map: Dict["VirtualSiteKey", int] = Field(
        dict(),
        description="A mapping between VirtualSiteKey objects (stored analogously to TopologyKey objects"
        "in other handlers) and topology indices describing the associated virtual site",
    )
    exclusion_policy: Literal[
        "none", "minimal", "parents", "local", "neighbors", "connected", "all"
    ] = "parents"

    @classmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        return [VirtualSiteHandler]

    @classmethod
    def supported_parameters(cls):
        """Return a list of parameter attributes supported by this handler."""
        return [
            "type",
            "name",
            "id",
            "match",
            "smirks",
            "sigma",
            "epsilon",
            "rmin_half",
            "charge_increment",
            "distance",
            "outOfPlaneAngle",
            "inPlaneAngle",
        ]

    def store_matches(
        self,
        parameter_handler: ParameterHandler,
        topology: "Topology",
    ) -> None:
        """Populate self.slot_map with key-val pairs of [VirtualSiteKey, PotentialKey]."""
        if self.slot_map:
            self.slot_map = dict()

        # Initialze the virtual site index to begin after the topoogy's atoms (0-indexed)
        virtual_site_index = topology.n_atoms

        matches_by_parent = parameter_handler._find_matches_by_parent(topology)

        for parent_index, parameters in matches_by_parent.items():
            for parameter, orientations in parameters:
                for orientation in orientations:

                    orientation_indices = orientation.topology_atom_indices

                    virtual_site_key = VirtualSiteKey(
                        parent_atom_index=parent_index,
                        orientation_atom_indices=orientation_indices,
                        type=parameter.type,
                        name=parameter.name,
                        match=parameter.match,
                    )

                    # TODO: Better way of specifying unique parameters
                    potential_key = PotentialKey(
                        id=" ".join(
                            [parameter.smirks, parameter.name, parameter.match]
                        ),
                        associated_handler="VirtualSites",
                    )
                    self.slot_map[virtual_site_key] = potential_key
                    self.virtual_site_key_topology_index_map[
                        virtual_site_key
                    ] = virtual_site_index
                    virtual_site_index += 1

    def store_potentials(  # type: ignore[override]
        self,
        parameter_handler: VirtualSiteHandler,
        vdw_handler: SMIRNOFFvdWHandler,
        electrostatics_handler: SMIRNOFFElectrostaticsHandler,
    ) -> None:
        """Store VirtualSite-specific parameter-like data."""
        if self.potentials:
            self.potentials = dict()
        for virtual_site_key, potential_key in self.slot_map.items():
            # TODO: This logic assumes no spaces in the SMIRKS pattern, name or `match` attribute
            smirks, _, _ = potential_key.id.split(" ")
            parameter = parameter_handler.get_parameter({"smirks": smirks})[0]

            virtual_site_potential = Potential(
                parameters={
                    "distance": parameter.distance,
                },
            )
            for attr in ["outOfPlaneAngle", "inPlaneAngle"]:
                if hasattr(parameter, attr):
                    virtual_site_potential.parameters.update(
                        {attr: getattr(parameter, attr)}
                    )
            self.potentials[potential_key] = virtual_site_potential

            vdw_key = PotentialKey(id=potential_key.id, associated_handler="vdw")
            vdw_potential = Potential(
                parameters={
                    "sigma": _compute_lj_sigma(parameter.sigma, parameter.rmin_half),
                    "epsilon": parameter.epsilon,
                },
            )
            vdw_handler.slot_map[virtual_site_key] = vdw_key  # type: ignore[index]
            vdw_handler.potentials[vdw_key] = vdw_potential

            electrostatics_key = PotentialKey(
                id=potential_key.id, associated_handler="Electrostatics"
            )
            electrostatics_potential = Potential(
                parameters={
                    "charge_increments": _validated_list_to_array(
                        parameter.charge_increment
                    ),
                }
            )
            electrostatics_handler.slot_map[virtual_site_key] = electrostatics_key  # type: ignore[index]
            electrostatics_handler.potentials[
                electrostatics_key
            ] = electrostatics_potential

    def _get_local_frame_weights(self, virtual_site_key: "VirtualSiteKey"):
        if virtual_site_key.type == "BondCharge":
            origin_weight = [1.0, 0.0]
            x_direction = [-1.0, 1.0]
            y_direction = [-1.0, 1.0]
        elif virtual_site_key.type == "MonovalentLonePair":
            origin_weight = [1, 0.0, 0.0]
            x_direction = [-1.0, 1.0, 0.0]
            y_direction = [-1.0, 0.0, 1.0]
        elif virtual_site_key.type == "DivalentLonePair":
            origin_weight = [0.0, 1.0, 0.0]
            x_direction = [0.5, -1.0, 0.5]
            y_direction = [1.0, -1.0, 0.0]
        elif virtual_site_key.type == "TrivalentLonePair":
            origin_weight = [0.0, 1.0, 0.0, 0.0]
            x_direction = [1 / 3, -1.0, 1 / 3, 1 / 3]
            y_direction = [1.0, -1.0, 0.0, 0.0]

        return origin_weight, x_direction, y_direction

    def _get_local_frame_position(self, virtual_site_key: "VirtualSiteKey"):
        potential_key = self.slot_map[virtual_site_key]
        potential = self.potentials[potential_key]
        if virtual_site_key.type == "BondCharge":
            distance = potential.parameters["distance"]
            local_frame_position = np.asarray([-1.0, 0.0, 0.0]) * distance
        elif virtual_site_key.type == "MonovalentLonePair":
            distance = potential.parameters["distance"]
            theta = potential.parameters["inPlaneAngle"].m_as(unit.radian)
            psi = potential.parameters["outOfPlaneAngle"].m_as(unit.radian)
            factor = np.array(
                [np.cos(theta) * np.cos(psi), np.sin(theta) * np.cos(psi), np.sin(psi)]
            )
            local_frame_position = factor * distance
        elif virtual_site_key.type == "DivalentLonePair":
            distance = potential.parameters["distance"]
            theta = potential.parameters["outOfPlaneAngle"].m_as(unit.radian)
            factor = np.asarray([-1.0 * np.cos(theta), 0.0, np.sin(theta)])
            local_frame_position = factor * distance
        elif virtual_site_key.type == "TrivalentLonePair":
            distance = potential.parameters["distance"]
            local_frame_position = np.asarray([-1.0, 0.0, 0.0]) * distance

        return local_frame_position


def library_charge_from_molecule(
    molecule: "Molecule",
) -> LibraryChargeHandler.LibraryChargeType:
    """Given an OpenFF Molecule with charges, generate a corresponding LibraryChargeType."""
    if molecule.partial_charges is None:
        raise ValueError("Input molecule is missing partial charges.")

    smirks = molecule.to_smiles(mapped=True)
    charges = molecule.partial_charges

    library_charge_type = LibraryChargeHandler.LibraryChargeType(
        smirks=smirks, charge=charges
    )

    return library_charge_type


def _get_interpolation_coeffs(fractional_bond_order, data):
    x1, x2 = data.keys()
    coeff1 = (x2 - fractional_bond_order) / (x2 - x1)
    coeff2 = (fractional_bond_order - x1) / (x2 - x1)

    return coeff1, coeff2


def _check_partial_bond_orders(
    reference_molecule: Molecule, molecule_list: List[Molecule]
) -> bool:
    """Check if the reference molecule is isomorphic with any molecules in a provided list."""
    if molecule_list is None:
        return False

    if len(molecule_list) == 0:
        return False

    for molecule in molecule_list:
        if reference_molecule.is_isomorphic_with(molecule):
            # TODO: Here is where a check for "all bonds in this molecule must have partial bond orders assigned"
            #       would go. That seems like a difficult mangled state to end up in, so not implemented for now.
            return True

    return False


SMIRNOFF_POTENTIAL_HANDLERS = [
    SMIRNOFFBondHandler,
    SMIRNOFFConstraintHandler,
    SMIRNOFFAngleHandler,
    SMIRNOFFProperTorsionHandler,
    SMIRNOFFImproperTorsionHandler,
    SMIRNOFFvdWHandler,
    SMIRNOFFElectrostaticsHandler,
    SMIRNOFFVirtualSiteHandler,
]


def _upconvert_bondhandler(bond_handler: BondHandler):
    """Given a BondHandler with version 0.3, up-convert to 0.4."""
    from packaging.version import Version

    assert bond_handler.version == Version(
        "0.3"
    ), "This up-converter only works with version 0.3."

    bond_handler.version = Version("0.4")
    bond_handler.potential = "(k/2)*(r-length)^2"


def _slow_key_lookup_by_atom_index(matches: Dict, atom_index: int) -> List[TopologyKey]:
    matched_keys = list()
    for key in matches:
        if (getattr(key, "this_atom_index", None) == atom_index) or (
            getattr(key, "atom_indices", [None])[0] == atom_index
        ):
            matched_keys.append(key)
    return matched_keys


def _compute_lj_sigma(
    sigma: Optional[unit.Quantity], rmin_half: Optional[unit.Quantity]
) -> unit.Quantity:

    return sigma if sigma is not None else (2.0 * rmin_half / (2.0 ** (1.0 / 6.0)))  # type: ignore


# Coped from the toolkit, see
# https://github.com/openforcefield/openff-toolkit/blob/0133414d3ab51e1af0996bcebe0cc1bdddc6431b/
# openff/toolkit/typing/engines/smirnoff/forcefield.py#L609
# However, it's not clear it's being called by any toolkit methods (in 0.10.x, 0.11.x, or at any point in history):
# https://github.com/openforcefield/openff-toolkit/search?q=_check_for_missing_valence_terms&type=code
def _check_for_missing_valence_terms(name, topology, assigned_terms, topological_terms):
    """Check that there are no missing valence terms in the given topology."""
    # Convert assigned terms and topological terms to lists
    assigned_terms = [item for item in assigned_terms]
    topological_terms = [item for item in topological_terms]

    def ordered_tuple(atoms):
        atoms = list(atoms)
        if atoms[0] < atoms[-1]:
            return tuple(atoms)
        else:
            return tuple(reversed(atoms))

    try:
        topology_set = {
            ordered_tuple(atom.index for atom in atomset)
            for atomset in topological_terms
        }
        assigned_set = {
            ordered_tuple(index for index in atomset) for atomset in assigned_terms
        }
    except TypeError:
        topology_set = {atom.index for atom in topological_terms}
        assigned_set = {atomset[0] for atomset in assigned_terms}

    def render_atoms(atomsets):
        msg = ""
        for atomset in atomsets:
            msg += f"{atomset:30} :"
            try:
                for atom_index in atomset:
                    atom = atoms[atom_index]
                    msg += (
                        f" {atom.residue.index:5} {atom.residue.name:3} {atom.name:3}"
                    )
            except TypeError:
                atom = atoms[atomset]
                msg += f" {atom.residue.index:5} {atom.residue.name:3} {atom.name:3}"

            msg += "\n"
        return msg

    if set(assigned_set) != set(topology_set):
        # Form informative error message
        msg = f"{name}: Mismatch between valence terms added and topological terms expected.\n"
        atoms = [atom for atom in topology.atoms]
        if len(assigned_set.difference(topology_set)) > 0:
            msg += "Valence terms created that are not present in Topology:\n"
            msg += render_atoms(assigned_set.difference(topology_set))
        if len(topology_set.difference(assigned_set)) > 0:
            msg += "Topological atom sets not assigned parameters:\n"
            msg += render_atoms(topology_set.difference(assigned_set))
        msg += "topology_set:\n"
        msg += str(topology_set) + "\n"
        msg += "assigned_set:\n"
        msg += str(assigned_set) + "\n"
        # TODO: Raise a more specific exception or delete this method
        raise Exception(msg)


# Coped from the toolkit, see
# https://github.com/openforcefield/openff-toolkit/blob/0133414d3ab51e1af0996bcebe0cc1bdddc6431b/
# openff/toolkit/typing/engines/smirnoff/parameters.py#L2318
def _check_all_valence_terms_assigned(
    handler,
    assigned_terms,
    topology,
    valence_terms,
    exception_cls=UnassignedValenceParameterException,
):
    """Check that all valence terms have been assigned."""
    if len(assigned_terms) == len(valence_terms):
        return

    # Convert the valence term to a valence dictionary to make sure
    # the order of atom indices doesn't matter for comparison.
    valence_terms_dict = assigned_terms.__class__()
    for atoms in valence_terms:
        atom_indices = (topology.atom_index(a) for a in atoms)
        valence_terms_dict[atom_indices] = atoms

    # Check that both valence dictionaries have the same keys (i.e. terms).
    assigned_terms_set = set(assigned_terms.keys())
    valence_terms_set = set(valence_terms_dict.keys())
    unassigned_terms = valence_terms_set.difference(assigned_terms_set)
    not_found_terms = assigned_terms_set.difference(valence_terms_set)

    # Raise an error if there are unassigned terms.
    err_msg = ""

    if len(unassigned_terms) > 0:

        unassigned_atom_tuples = []

        unassigned_str = ""
        for unassigned_tuple in unassigned_terms:
            unassigned_str += "\n- Topology indices " + str(unassigned_tuple)
            unassigned_str += ": names and elements "

            unassigned_atoms = []

            # Pull and add additional helpful info on missing terms
            for atom_idx in unassigned_tuple:
                atom = topology.atom(atom_idx)
                unassigned_atoms.append(atom)
                unassigned_str += f"({atom.name} {atom.symbol}), "
            unassigned_atom_tuples.append(tuple(unassigned_atoms))
        err_msg += (
            "{parameter_handler} was not able to find parameters for the following valence terms:\n"
            "{unassigned_str}"
        ).format(
            parameter_handler=handler.__class__.__name__, unassigned_str=unassigned_str
        )
    if len(not_found_terms) > 0:
        if err_msg != "":
            err_msg += "\n"
        not_found_str = "\n- ".join([str(x) for x in not_found_terms])
        err_msg += (
            "{parameter_handler} assigned terms that were not found in the topology:\n"
            "- {not_found_str}"
        ).format(
            parameter_handler=handler.__class__.__name__, not_found_str=not_found_str
        )
    if err_msg != "":
        err_msg += "\n"
        exception = exception_cls(err_msg)
        exception.unassigned_topology_atom_tuples = unassigned_atom_tuples
        exception.handler_class = handler.__class__
        raise exception
