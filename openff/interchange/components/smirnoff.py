"""Models and utilities for processing SMIRNOFF data."""
import abc
import copy
import functools
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
from openff.interchange.exceptions import (
    InvalidParameterHandlerError,
    MissingParametersError,
    SMIRNOFFParameterAttributeNotImplementedError,
)
from openff.interchange.models import PotentialKey, TopologyKey, VirtualSiteKey
from openff.interchange.types import FloatQuantity

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


class SMIRNOFFPotentialHandler(PotentialHandler, abc.ABC):
    """Base class for handlers storing potentials produced by SMIRNOFF force fields."""

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

            parameter_handler._check_all_valence_terms_assigned(
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


class SMIRNOFFBondHandler(SMIRNOFFPotentialHandler):
    """Handler storing bond potentials as produced by a SMIRNOFF force field."""

    type: Literal["Bonds"] = "Bonds"
    expression: Literal["k/2*(r-length)**2"] = "k/2*(r-length)**2"
    fractional_bond_order_method: Literal["AM1-Wiberg", "None"] = "AM1-Wiberg"
    fractional_bond_order_interpolation: Literal["linear"] = "linear"

    # Note that Parsley shipped with `"None"` (not `None`!) as the default value
    # for the bond order interpolation, so disallowing it would be problematic.
    #
    # >>> from openff.toolkit.typing.engines.smirnoff import ForceField
    # >>> ForceField("openff-1.0.0.offxml")['Bonds'].fractional_bondorder_method
    # 'None'
    # >>> ForceField("openff-1.0.0.offxml")['ProperTorsions'].fractional_bondorder_method
    # 'AM1-Wiberg'

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

        parameter_handler._check_all_valence_terms_assigned(
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
            if topology_key.bond_order:  # type: ignore[union-attr]
                bond_order = topology_key.bond_order  # type: ignore[union-attr]
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

    @classmethod
    def f_from_toolkit(
        cls: Type[T],
        parameter_handler: "AngleHandler",
        topology: "Topology",
    ) -> T:
        """
        Create a SMIRNOFFAngleHandler from toolkit data.

        """
        handler = cls()
        handler.store_matches(parameter_handler=parameter_handler, topology=topology)
        handler.store_potentials(parameter_handler=parameter_handler)

        return handler


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

        parameter_handler._check_all_valence_terms_assigned(
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
            if topology_key.bond_order:  # type: ignore[union-attr]
                bond_order = topology_key.bond_order  # type: ignore[union-attr]
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
        for potential_key in self.slot_map.values():
            smirks = potential_key.id
            n = potential_key.mult
            parameter = parameter_handler.get_parameter({"smirks": smirks})[0]
            parameters = {
                "k": parameter.k[n],
                "periodicity": parameter.periodicity[n] * unit.dimensionless,
                "phase": parameter.phase[n],
                "idivf": 3.0 * unit.dimensionless,
            }
            potential = Potential(parameters=parameters)
            self.potentials[potential_key] = potential


class _SMIRNOFFNonbondedHandler(SMIRNOFFPotentialHandler, abc.ABC):
    """Base class for handlers storing non-bonded potentials produced by SMIRNOFF force fields."""

    type: Literal["nonbonded"] = "nonbonded"

    cutoff: FloatQuantity["angstrom"] = Field(  # type: ignore
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

    type: Literal["vdW"] = "vdW"  # type: ignore[assignment]

    expression: Literal[
        "4*epsilon*((sigma/r)**12-(sigma/r)**6)"
    ] = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"

    method: Literal["cutoff", "pme", "no-cutoff"] = Field("cutoff")

    mixing_rule: Literal["lorentz-berthelot", "geometric"] = Field(
        "lorentz-berthelot",
        description="The mixing rule (combination rule) used in computing pairwise vdW interactions",
    )

    switch_width: FloatQuantity["angstrom"] = Field(  # type: ignore
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

            self.slot_map.update({top_key: pot_key})
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

    type: Literal["Electrostatics"] = "Electrostatics"  # type: ignore[assignment]
    expression: Literal["coul"] = "coul"

    method: Literal["pme", "cutoff", "reaction-field", "no-cutoff"] = Field("pme")

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
                    charge = -1.0 * np.sum(parameter_value)
                    # assumes virtual sites can only have charges determined in one step
                    # also, topology_key is actually a VirtualSiteKey
                    charges[topology_key] = charge
                elif parameter_key in ["charge", "charge_increment"]:
                    charge = parameter_value
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
    ) -> T:
        """
        Create a SMIRNOFFElectrostaticsHandler from toolkit data.

        """
        if isinstance(parameter_handler, list):
            parameter_handlers = parameter_handler
        else:
            parameter_handlers = [parameter_handler]

        toolkit_handler_with_metadata = [
            p for p in parameter_handlers if type(p) == ElectrostaticsHandler
        ][0]

        handler = cls(
            type=toolkit_handler_with_metadata._TAGNAME,
            scale_13=toolkit_handler_with_metadata.scale13,
            scale_14=toolkit_handler_with_metadata.scale14,
            scale_15=toolkit_handler_with_metadata.scale15,
            cutoff=toolkit_handler_with_metadata.cutoff,
            method=toolkit_handler_with_metadata.method.lower(),
        )

        handler.store_matches(
            parameter_handlers,
            topology,
            charge_from_molecules=charge_from_molecules,  # type: ignore[call-arg]
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

            self.slot_map.update({virtual_site_key: virtual_site_potential_key})
            self.potentials.update({virtual_site_potential_key: virtual_site_potential})

            for i, atom_index in enumerate(atom_indices):  # noqa
                topology_key = TopologyKey(atom_indices=(atom_index,), mult=i)

                # TODO: Better way of dedupliciating this case (charge increments from multiple different
                #       virtual sites are applied to the same atom)
                while topology_key in self.slot_map:
                    topology_key.mult += 1000

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
            topology_key = TopologyKey(atom_indices=(atom_index,))
            potential_key = PotentialKey(
                id=parameter.smirks, mult=i, associated_handler="LibraryCharges"
            )
            potential = Potential(parameters={"charge": charge})

            matches[topology_key] = potential_key
            potentials[potential_key] = potential

        return matches, potentials

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
            topology_key = TopologyKey(atom_indices=(atom_index,))
            potential_key = PotentialKey(
                id=parameter.smirks, mult=i, associated_handler="ChargeIncrementModel"
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

            matches.update(parameter_matches)
            potentials.update(parameter_potentials)

        return matches, potentials

    @classmethod
    def _find_am1_matches(
        cls,
        parameter_handler: Union["ToolkitAM1BCCHandler", ChargeIncrementModelHandler],
        unique_molecule: Molecule,
    ) -> Tuple[Dict[TopologyKey, PotentialKey], Dict[PotentialKey, Potential]]:
        """Construct a slot and potential map for a charge model based parameter handler."""
        unique_molecule = copy.deepcopy(unique_molecule)
        reference_smiles = unique_molecule.to_smiles(
            isomeric=True, explicit_hydrogens=True, mapped=True
        )

        if isinstance(parameter_handler, ChargeIncrementModelHandler):
            partial_charge_method = parameter_handler.partial_charge_method
        elif isinstance(parameter_handler, ToolkitAM1BCCHandler):
            # TODO: There needs to be a cleaner way of doing this check, since it's not
            #       exposed as an attribute of ToolkitAM1BCCHandler and the check that the
            #       toolkit does is internal to that handler. Implementation at
            #       https://github.com/openforcefield/openff-toolkit/blob/0c42148bcbd984af50236696ad281c98cf6d8a0a/openff/toolkit/typing/engines/smirnoff/parameters.py#L4198-L4210
            try:
                from openeye import oechem

                if oechem.OEChemIsLicensed():
                    partial_charge_method = "am1bccelf10"
                else:
                    partial_charge_method = "am1bcc"
            except ImportError:
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

        for i, partial_charge in enumerate(partial_charges):

            potential_key = PotentialKey(
                id=reference_smiles, mult=i, associated_handler="ToolkitAM1BCC"
            )
            potentials[potential_key] = Potential(parameters={"charge": partial_charge})

            matches[TopologyKey(atom_indices=(i,))] = potential_key

        return matches, potentials

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

                am1_matches, am1_potentials = cls._find_am1_matches(
                    parameter_handler,
                    unique_molecule,
                )

            if slot_matches is None and am1_matches is None:
                raise NotImplementedError()

            elif slot_matches is not None and am1_matches is not None:

                am1_matches = {
                    TopologyKey(
                        atom_indices=topology_key.atom_indices, mult=0
                    ): potential_key
                    for topology_key, potential_key in am1_matches.items()
                }
                slot_matches = {
                    TopologyKey(
                        atom_indices=topology_key.atom_indices, mult=1
                    ): potential_key
                    for topology_key, potential_key in slot_matches.items()
                }

                matched_atom_indices = {
                    index for key in slot_matches for index in key.atom_indices
                }
                matched_atom_indices.intersection_update(
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
                f"not be fully assigned charges."
            )

        return matches, potentials

    @classmethod
    def _assign_charges_from_charge_from_molecules(
        cls,
        topology: "Topology",
        unique_molecule: Molecule,
        charge_from_molecules=Optional[List[Molecule]],
    ) -> Tuple[bool, Dict, Dict]:
        if charge_from_molecules is None:
            return False, dict(), dict()

        for reference_molecule in charge_from_molecules:
            if reference_molecule.is_isomorphic_with(unique_molecule):
                break
        else:
            return False, dict(), dict()

        matches = dict()
        potentials = dict()
        mapped_smiles = unique_molecule.to_smiles(mapped=True, explicit_hydrogens=True)

        for index, partial_charge in enumerate(reference_molecule.partial_charges):
            topology_key = TopologyKey(atom_indices=(index,))
            potential_key = PotentialKey(
                id=mapped_smiles,
                mult=index,
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

            flag, matches, potentials = self._assign_charges_from_charge_from_molecules(
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

            match_mults = defaultdict(set)

            for top_key in matches:
                match_mults[top_key.atom_indices].add(top_key.mult)

            self.potentials.update(potentials)

            for unique_molecule_particle in unique_molecule.particles:
                unique_molecule_particle_index = unique_molecule.particle_index(
                    unique_molecule_particle
                )
                # particle_charge = unique_molecule.partial_charges[unique_molecule_particle_index]

                for duplicate_molecule_index, atom_map in group:
                    duplicate_molecule = topology.molecule(duplicate_molecule_index)
                    duplicate_molecule_particle_index = atom_map[
                        unique_molecule_particle_index
                    ]
                    duplicate_molecule_particle = duplicate_molecule.particle(
                        duplicate_molecule_particle_index
                    )
                    topology_particle_index = topology.particle_index(
                        duplicate_molecule_particle
                    )

                    for mult in match_mults[(unique_molecule_particle_index,)]:
                        topology_key = TopologyKey(
                            atom_indices=(topology_particle_index,),
                            mult=mult,
                        )
                        reference_key = TopologyKey(
                            atom_indices=(unique_molecule_particle_index,), mult=mult
                        )
                        potential_key = matches[reference_key]

                        self.slot_map[topology_key] = potential_key

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
        """
        Populate self.slot_map with key-val pairs of [TopologyKey, PotentialKey].

        Differs from SMIRNOFFPotentialHandler.store_matches because each key
        can point to multiple potentials (?); each value in the dict is a
        list of parametertypes, whereas conventional handlers don't have lists
        """
        virtual_site_index = topology.n_atoms
        parameter_handler_name = getattr(parameter_handler, "_TAGNAME", None)
        if self.slot_map:
            self.slot_map = dict()
        matches = parameter_handler.find_matches(topology)
        for key, val_list in matches.items():
            for val in val_list:
                virtual_site_key = VirtualSiteKey(
                    atom_indices=key,
                    type=val.parameter_type.type,
                    name=val.parameter_type.name,
                    match=val.parameter_type.match,
                )
                potential_key = PotentialKey(
                    id=val.parameter_type.smirks,
                    associated_handler=parameter_handler_name,
                )
                self.slot_map[virtual_site_key] = potential_key
                self.virtual_site_key_topology_index_map[
                    virtual_site_key
                ] = virtual_site_index
                virtual_site_index += 1

    def store_potentials(self, parameter_handler: ParameterHandler) -> None:
        """Store VirtualSite-specific parameter-like data."""
        if self.potentials:
            self.potentials = dict()
        for potential_key in self.slot_map.values():
            smirks = potential_key.id
            parameter_type = parameter_handler.get_parameter({"smirks": smirks})[0]
            potential = Potential(
                parameters={
                    "distance": parameter_type.distance,
                },
            )
            for attr in ["outOfPlaneAngle", "inPlaneAngle"]:
                if hasattr(parameter_type, attr):
                    potential.parameters.update({attr: getattr(parameter_type, attr)})
            self.potentials[potential_key] = potential

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
            theta = potential.parameters["inPlaneAngle"].m_as(unit.radian)  # type: ignore
            psi = potential.parameters["outOfPlaneAngle"].m_as(unit.radian)  # type: ignore
            factor = np.array(
                [np.cos(theta) * np.cos(psi), np.sin(theta) * np.cos(psi), np.sin(psi)]
            )
            local_frame_position = factor * distance
        elif virtual_site_key.type == "DivalentLonePair":
            distance = potential.parameters["distance"]
            theta = potential.parameters["outOfPlaneAngle"].m_as(unit.radian)  # type: ignore
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
