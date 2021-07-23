import abc
import copy
import functools
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, TypeVar, Union

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
    VirtualSiteHandler,
    vdWHandler,
)
from openff.units import unit
from openff.units.simtk import from_simtk
from pydantic import Field
from simtk import unit as omm_unit
from typing_extensions import Literal

from openff.interchange.components.potentials import Potential, PotentialHandler
from openff.interchange.exceptions import (
    InvalidParameterHandlerError,
    SMIRNOFFParameterAttributeNotImplementedError,
)
from openff.interchange.models import PotentialKey, TopologyKey, VirtualSiteKey
from openff.interchange.types import FloatQuantity

kcal_mol = omm_unit.kilocalorie_per_mole
kcal_mol_angstroms = kcal_mol / omm_unit.angstrom ** 2
kcal_mol_radians = kcal_mol / omm_unit.radian ** 2

if TYPE_CHECKING:
    from openff.toolkit.topology import Topology

    from openff.interchange.components.mdtraj import OFFBioTop

    ElectrostaticsHandlerType = Union[
        ElectrostaticsHandler,
        ChargeIncrementModelHandler,
        LibraryChargeHandler,
        ToolkitAM1BCCHandler,
    ]


T = TypeVar("T", bound="SMIRNOFFPotentialHandler")
T_ = TypeVar("T_", bound="PotentialHandler")


class SMIRNOFFPotentialHandler(PotentialHandler, abc.ABC):
    @classmethod
    @abc.abstractmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes (toolkit)"""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def supported_parameters(cls):
        """Return a list of parameter attributes supported by this handler"""
        raise NotImplementedError()

    @classmethod
    def check_supported_parameters(cls, parameter_handler: ParameterHandler):
        for parameter in parameter_handler.parameters:
            for parameter_attribute in parameter._get_defined_parameter_attributes():
                if parameter_attribute not in cls.supported_parameters():
                    raise SMIRNOFFParameterAttributeNotImplementedError(
                        parameter_attribute,
                    )

    def store_matches(
        self,
        parameter_handler: ParameterHandler,
        topology: Union["Topology", "OFFBioTop"],
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
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

    @classmethod
    def _from_toolkit(
        cls: Type[T],
        parameter_handler: T_,
        topology: "Topology",
    ) -> T:
        """
        Create a SMIRNOFFPotentialHandler from toolkit data.

        """
        if type(parameter_handler) not in cls.allowed_parameter_handlers():
            raise InvalidParameterHandlerError(type(parameter_handler))

        handler = cls()
        handler.store_matches(parameter_handler=parameter_handler, topology=topology)
        handler.store_potentials(parameter_handler=parameter_handler)

        return handler


class SMIRNOFFBondHandler(SMIRNOFFPotentialHandler):

    type: Literal["Bonds"] = "Bonds"
    expression: Literal["k/2*(r-length)**2"] = "k/2*(r-length)**2"

    @classmethod
    def allowed_parameter_handlers(cls):
        return [BondHandler]

    @classmethod
    def supported_parameters(cls):
        return ["smirks", "id", "k", "length"]

    def store_potentials(self, parameter_handler: "BondHandler") -> None:
        """
        Populate self.potentials with key-val pairs of unique potential
        identifiers and their associated Potential objects

        """
        if self.potentials:
            self.potentials = dict()
        for potential_key in self.slot_map.values():
            smirks = potential_key.id
            parameter_type = parameter_handler.get_parameter({"smirks": smirks})[0]
            potential = Potential(
                parameters={
                    "k": parameter_type.k,
                    "length": parameter_type.length,
                },
            )
            self.potentials[potential_key] = potential

    @classmethod
    def _from_toolkit(  # type: ignore[override]
        cls: Type[T],
        parameter_handler: "BondHandler",
        topology: "Topology",
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

        handler: T = cls(type="Bonds", expression="k/2*(r-length)**2")
        handler.store_matches(parameter_handler=parameter_handler, topology=topology)
        handler.store_potentials(parameter_handler=parameter_handler)

        return handler


class SMIRNOFFConstraintHandler(SMIRNOFFPotentialHandler):

    type: Literal["Constraints"] = "Constraints"
    expression: Literal[""] = ""
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    constraints: Dict[
        PotentialKey, bool
    ] = dict()  # should this be named potentials for consistency?

    @classmethod
    def allowed_parameter_handlers(cls):
        return [BondHandler, ConstraintHandler]

    @classmethod
    def supported_parameters(cls):
        return ["smirks", "id", "k", "length", "distance"]

    @classmethod
    def _from_toolkit(
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
        handler.store_constraints(
            parameter_handlers=parameter_handlers, topology=topology
        )

        return handler

    def store_constraints(
        self,
        parameter_handlers: Any,
        topology: "OFFBioTop",
    ) -> None:

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
                distance = match.parameter_type.distance
            else:
                # This constraint parameter depends on the BondHandler ...
                if bond_handler is None:
                    from openff.interchange.exceptions import MissingParametersError

                    raise MissingParametersError(
                        f"Constraint with SMIRKS pattern {smirks} found with no distance "
                        "specified, and no corresponding bond parameters were found. The distance "
                        "of this constraint is not specified."
                    )
                # ... so use the same PotentialKey instance as the BondHandler to look up the distance
                potential_key = bonds.slot_map[topology_key]
                self.slot_map[topology_key] = potential_key
                distance = bonds.potentials[potential_key].parameters["length"]
            potential = Potential(
                parameters={
                    "distance": distance,
                }
            )
            self.constraints[potential_key] = potential  # type: ignore[assignment]


class SMIRNOFFAngleHandler(SMIRNOFFPotentialHandler):

    type: Literal["Angles"] = "Angles"
    expression: Literal["k/2*(theta-angle)**2"] = "k/2*(theta-angle)**2"

    @classmethod
    def allowed_parameter_handlers(cls):
        return [AngleHandler]

    @classmethod
    def supported_parameters(cls):
        return ["smirks", "id", "k", "angle"]

    def store_potentials(self, parameter_handler: "AngleHandler") -> None:
        """
        Populate self.potentials with key-val pairs of unique potential
        identifiers and their associated Potential objects

        """
        for potential_key in self.slot_map.values():
            smirks = potential_key.id
            # ParameterHandler.get_parameter returns a list, although this
            # should only ever be length 1
            parameter_type = parameter_handler.get_parameter({"smirks": smirks})[0]
            potential = Potential(
                parameters={
                    "k": parameter_type.k,
                    "angle": parameter_type.angle,
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

    type: Literal["ProperTorsions"] = "ProperTorsions"
    expression: Literal[
        "k*(1+cos(periodicity*theta-phase))"
    ] = "k*(1+cos(periodicity*theta-phase))"

    @classmethod
    def allowed_parameter_handlers(cls):
        return [ProperTorsionHandler]

    @classmethod
    def supported_parameters(cls):
        return ["smirks", "id", "k", "periodicity", "phase", "idivf"]

    def store_matches(
        self,
        parameter_handler: "ProperTorsionHandler",
        topology: "OFFBioTop",
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
        if self.slot_map:
            self.slot_map = dict()
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            n_terms = len(val.parameter_type.k)
            for n in range(n_terms):
                smirks = val.parameter_type.smirks
                topology_key = TopologyKey(atom_indices=key, mult=n)
                potential_key = PotentialKey(
                    id=smirks, mult=n, associated_handler="ProperTorsions"
                )
                self.slot_map[topology_key] = potential_key

    def store_potentials(self, parameter_handler: "ProperTorsionHandler") -> None:
        """
        Populate self.potentials with key-val pairs of unique potential
        identifiers and their associated Potential objects

        """
        for potential_key in self.slot_map.values():
            smirks = potential_key.id
            n = potential_key.mult
            parameter_type = parameter_handler.get_parameter({"smirks": smirks})[0]
            # n_terms = len(parameter_type.k)
            parameters = {
                "k": parameter_type.k[n],
                "periodicity": parameter_type.periodicity[n] * unit.dimensionless,
                "phase": parameter_type.phase[n],
                "idivf": parameter_type.idivf[n] * unit.dimensionless,
            }
            potential = Potential(parameters=parameters)
            self.potentials[potential_key] = potential


class SMIRNOFFImproperTorsionHandler(SMIRNOFFPotentialHandler):

    type: Literal["ImproperTorsions"] = "ImproperTorsions"
    expression: Literal[
        "k*(1+cos(periodicity*theta-phase))"
    ] = "k*(1+cos(periodicity*theta-phase))"

    @classmethod
    def allowed_parameter_handlers(cls):
        return [ImproperTorsionHandler]

    @classmethod
    def supported_parameters(cls):
        return ["smirks", "id", "k", "periodicity", "phase", "idivf"]

    def store_matches(
        self, parameter_handler: "ImproperTorsionHandler", topology: "OFFBioTop"
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

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
        Populate self.potentials with key-val pairs of unique potential
        identifiers and their associated Potential objects

        """
        for potential_key in self.slot_map.values():
            smirks = potential_key.id
            n = potential_key.mult
            parameter_type = parameter_handler.get_parameter({"smirks": smirks})[0]
            parameters = {
                "k": parameter_type.k[n],
                "periodicity": parameter_type.periodicity[n] * unit.dimensionless,
                "phase": parameter_type.phase[n],
                "idivf": 3.0 * unit.dimensionless,
            }
            potential = Potential(parameters=parameters)
            self.potentials[potential_key] = potential


class _SMIRNOFFNonbondedHandler(SMIRNOFFPotentialHandler, abc.ABC):
    """The base class for handlers which store nonbonded potentials."""

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

    type: Literal["vdW"] = "vdW"

    expression: Literal[
        "4*epsilon*((sigma/r)**12-(sigma/r)**6)"
    ] = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"

    method: Literal["cutoff", "pme"] = Field("cutoff")

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
        return [vdWHandler]

    @classmethod
    def supported_parameters(cls):
        return ["smirks", "id", "sigma", "epsilon", "rmin_half"]

    def store_potentials(self, parameter_handler: vdWHandler) -> None:
        """
        Populate self.potentials with key-val pairs of unique potential
        identifiers and their associated Potential objects

        """
        self.method = parameter_handler.method.lower()
        self.cutoff = parameter_handler.cutoff

        for potential_key in self.slot_map.values():
            smirks = potential_key.id
            parameter_type = parameter_handler.get_parameter({"smirks": smirks})[0]
            try:
                potential = Potential(
                    parameters={
                        "sigma": parameter_type.sigma,
                        "epsilon": parameter_type.epsilon,
                    },
                )
            except AttributeError:
                # Handle rmin_half pending https://github.com/openforcefield/openff-toolkit/pull/750
                potential = Potential(
                    parameters={
                        "sigma": parameter_type.sigma,
                        "epsilon": parameter_type.epsilon,
                    },
                )
            self.potentials[potential_key] = potential

    @classmethod
    def _from_toolkit(
        cls: Type[T],
        parameter_handler: "vdWHandler",
        topology: "Topology",
    ) -> "SMIRNOFFvdWHandler":
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
        """The order in which parameter handlers take precedence when computing the
        charges
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
    """A handler which stores any electrostatic parameters applied to a topology.

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

    method: Literal["pme", "cutoff", "reaction-field"] = Field("pme")

    @classmethod
    def allowed_parameter_handlers(cls):
        return [
            LibraryChargeHandler,
            ChargeIncrementModelHandler,
            ToolkitAM1BCCHandler,
            ElectrostaticsHandler,
        ]

    @classmethod
    def supported_parameters(cls):
        pass

    @property
    def charges(self):
        return self.get_charges(include_virtual_sites=False)

    @property
    def charges_with_virtual_sites(self):
        return self.get_charges(include_virtual_sites=True)

    def get_charges(
        self, include_virtual_sites=False
    ) -> Dict[TopologyKey, unit.Quantity]:
        """Returns the total partial charge on each particle in the associated interchange."""

        charges = defaultdict(lambda: 0.0 * unit.e)

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
                    charges[topology_key.atom_indices[0]] += charge
                else:
                    raise NotImplementedError()

        returned_charges = {}

        for index, charge in charges.items():
            if isinstance(index, int):
                returned_charges[TopologyKey(atom_indices=(index,))] = charge
            else:
                if include_virtual_sites:
                    returned_charges[index] = charge

        return returned_charges

    @classmethod
    def parameter_handler_precedence(cls) -> List[str]:
        """The order in which parameter handlers take precedence when computing the
        charges
        """
        return ["LibraryCharges", "ChargeIncrementModel", "ToolkitAM1BCC"]

    @classmethod
    def _from_toolkit(
        cls: Type[T],
        parameter_handler: Any,
        topology: "Topology",
    ) -> "SMIRNOFFElectrostaticsHandler":
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

        handler.store_matches(parameter_handlers, topology)

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
                match=virtual_site_type.match,
            )

            virtual_site_potential_key = PotentialKey(
                id=virtual_site_type.smirks,
                associated_handler="VirtualSiteHandler",
            )

            virtual_site_potential = Potential(
                parameters={
                    "charge_increments": from_simtk(virtual_site_type.charge_increment),
                }
            )

            matches = {}
            potentials = {}

            self.slot_map.update({virtual_site_key: virtual_site_potential_key})
            self.potentials.update({virtual_site_potential_key: virtual_site_potential})

            # TODO: Counter-intuitive that toolkit regression tests pass by using the counter
            # variable i as if it was the atom index - shouldn't it just use atom_index?
            for i, atom_index in enumerate(atom_indices):  # noqa
                topology_key = TopologyKey(atom_indices=(i,), mult=2)
                potential_key = PotentialKey(
                    id=virtual_site_type.smirks,
                    mult=i,
                    associated_handler="VirtualSiteHandler",
                )

                charge_increment = getattr(
                    virtual_site_type, f"charge_increment{i + 1}"
                )

                potential = Potential(
                    parameters={"charge_increment": from_simtk(charge_increment)}
                )

                matches[topology_key] = potential_key
                potentials[potential_key] = potential

        self.slot_map.update(matches)
        self.potentials.update(potentials)

    @classmethod
    @functools.lru_cache(None)
    def _compute_partial_charges(cls, molecule: Molecule, method: str) -> unit.Quantity:
        """Call out to the toolkit's toolkit wrappers to generate partial charges"""
        molecule = copy.deepcopy(molecule)
        molecule.assign_partial_charges(method)

        return from_simtk(molecule.partial_charges)

    @classmethod
    def _library_charge_to_potentials(
        cls,
        atom_indices: Tuple[int, ...],
        parameter: LibraryChargeHandler.LibraryChargeType,
    ) -> Tuple[Dict[TopologyKey, PotentialKey], Dict[PotentialKey, Potential]]:
        """Maps a matched library charge parameter to a set of potentials."""

        matches = {}
        potentials = {}

        for i, (atom_index, charge) in enumerate(zip(atom_indices, parameter.charge)):
            topology_key = TopologyKey(atom_indices=(atom_index,))
            potential_key = PotentialKey(
                id=parameter.smirks, mult=i, associated_handler="LibraryCharges"
            )
            potential = Potential(parameters={"charge": from_simtk(charge)})

            matches[topology_key] = potential_key
            potentials[potential_key] = potential

        return matches, potentials

    @classmethod
    def _charge_increment_to_potentials(
        cls,
        atom_indices: Tuple[int, ...],
        parameter: ChargeIncrementModelHandler.ChargeIncrementType,
    ) -> Tuple[Dict[TopologyKey, PotentialKey], Dict[PotentialKey, Potential]]:
        """Maps a matched charge increment parameter to a set of potentials."""

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

            potential = Potential(
                parameters={"charge_increment": from_simtk(charge_increment)}
            )

            matches[topology_key] = potential_key
            potentials[potential_key] = potential

        return matches, potentials

    @classmethod
    def _find_slot_matches(
        cls,
        parameter_handler: Union["LibraryChargeHandler", "ChargeIncrementModelHandler"],
        reference_molecule: Molecule,
    ) -> Tuple[Dict[TopologyKey, PotentialKey], Dict[PotentialKey, Potential]]:
        """Constructs a slot and potential map for a slot based parameter handler."""

        # Ideally this would be made redundant by OpenFF TK #971
        unique_parameter_matches = {
            tuple(sorted(key)): (key, val)
            for key, val in parameter_handler.find_matches(
                reference_molecule.to_topology()
            ).items()
        }

        parameter_matches = {key: val for key, val in unique_parameter_matches.values()}

        matches, potentials = {}, {}

        for key, val in parameter_matches.items():

            parameter_type = val.parameter_type

            if isinstance(parameter_handler, LibraryChargeHandler):

                (
                    parameter_matches,
                    parameter_potentials,
                ) = cls._library_charge_to_potentials(key, parameter_type)

            elif isinstance(parameter_handler, ChargeIncrementModelHandler):

                (
                    parameter_matches,
                    parameter_potentials,
                ) = cls._charge_increment_to_potentials(key, parameter_type)

            else:
                raise NotImplementedError()

            matches.update(parameter_matches)
            potentials.update(parameter_potentials)

        return matches, potentials

    @classmethod
    def _find_am1_matches(
        cls,
        parameter_handler: Union["ToolkitAM1BCCHandler", ChargeIncrementModelHandler],
        reference_molecule: Molecule,
    ) -> Tuple[Dict[TopologyKey, PotentialKey], Dict[PotentialKey, Potential]]:
        """Constructs a slot and potential map for a charge model based parameter handler."""

        reference_molecule = copy.deepcopy(reference_molecule)
        reference_smiles = reference_molecule.to_smiles(
            isomeric=True, explicit_hydrogens=True, mapped=True
        )

        method = getattr(parameter_handler, "partial_charge_method", "am1bcc")

        partial_charges = cls._compute_partial_charges(
            reference_molecule, method=method
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
        reference_molecule: Molecule,
    ) -> Tuple[Dict[TopologyKey, PotentialKey], Dict[PotentialKey, Potential]]:
        """Constructs a slot and potential map for a particular reference molecule
        and set of parameter handlers."""

        matches = {}
        potentials = {}

        expected_matches = {i for i in range(reference_molecule.n_atoms)}

        for handler_type in cls.parameter_handler_precedence():

            if handler_type not in parameter_handlers:
                continue

            parameter_handler = parameter_handlers[handler_type]

            slot_matches, slot_potentials = None, {}
            am1_matches, am1_potentials = None, {}

            if handler_type in ["LibraryCharges", "ChargeIncrementModel"]:

                slot_matches, slot_potentials = cls._find_slot_matches(
                    parameter_handler, reference_molecule
                )

            if handler_type in ["ToolkitAM1BCC", "ChargeIncrementModel"]:

                am1_matches, am1_potentials = cls._find_am1_matches(
                    parameter_handler, reference_molecule
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
                    index for key in am1_matches for index in key.atom_indices
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
                f"{reference_molecule.to_smiles(explicit_hydrogens=False)} could "
                f"not be fully assigned charges."
            )

        return matches, potentials

    def store_matches(
        self,
        parameter_handler: Union[
            "ElectrostaticsHandlerType", List["ElectrostaticsHandlerType"]
        ],
        topology: Union["Topology", "OFFBioTop"],
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers
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

        reference_molecules = [*topology.reference_molecules]

        for reference_molecule in reference_molecules:

            matches, potentials = self._find_reference_matches(
                parameter_handlers, reference_molecule
            )

            match_mults = defaultdict(set)

            for top_key in matches:
                match_mults[top_key.atom_indices].add(top_key.mult)

            self.potentials.update(potentials)

            for top_mol in topology._reference_molecule_to_topology_molecules[
                reference_molecule
            ]:

                for topology_particle in top_mol.atoms:

                    reference_index = topology_particle.atom.molecule_particle_index
                    topology_index = topology_particle.topology_particle_index

                    for mult in match_mults[(reference_index,)]:

                        top_key = TopologyKey(atom_indices=(topology_index,), mult=mult)

                        self.slot_map[top_key] = matches[
                            TopologyKey(atom_indices=(reference_index,), mult=mult)
                        ]

    def store_potentials(
        self,
        parameter_handler: Union[
            "ElectrostaticsHandlerType", List["ElectrostaticsHandlerType"]
        ],
    ) -> None:
        # This logic is handled by ``store_matches`` as we may need to create potentials
        # to store depending on the handler type.
        pass


class SMIRNOFFVirtualSiteHandler(SMIRNOFFPotentialHandler):

    type: Literal["Bonds"] = "Bonds"
    expression: Literal[""] = ""

    @classmethod
    def allowed_parameter_handlers(cls):
        return [VirtualSiteHandler]

    @classmethod
    def supported_parameters(cls):
        return ["distance", "outOfPlaneAngle", "inPlaneAngle"]

    def store_matches(
        self,
        parameter_handler: ParameterHandler,
        topology: Union["Topology", "OFFBioTop"],
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        Differs from SMIRNOFFPotentialHandler.store_matches because each key
        can point to multiple potentials (?); each value in the dict is a
        list of parametertypes, whereas conventional handlers don't have lists
        """
        parameter_handler_name = getattr(parameter_handler, "_TAGNAME", None)
        if self.slot_map:
            self.slot_map = dict()
        matches = parameter_handler.find_matches(topology)
        for key, val_list in matches.items():
            for val in val_list:
                virtual_site_key = VirtualSiteKey(
                    atom_indices=key,
                    type=val.parameter_type.type,
                    match=val.parameter_type.match,
                )
                potential_key = PotentialKey(
                    id=val.parameter_type.smirks,
                    associated_handler=parameter_handler_name,
                )
                self.slot_map[virtual_site_key] = potential_key

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
                    potential.parameters.update(
                        {attr: from_simtk(getattr(parameter_type, attr))}
                    )
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
            y_direction = [1.0, -1.0, 1.0]
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
            local_frame_position = [-1.0, 0.0, 0.0] * distance
        elif virtual_site_key.type == "MonovalentLonePair":
            distance = potential.parameters["distance"]
            theta = potential.parameters["inPlaneAngle"].m_as(unit.radian)
            psi = potential.parameters["outOfPlaneAngle"].m_as(unit.radian)
            local_frame_position = [
                np.cos(theta) * np.cos(psi),
                np.sin(theta) * np.cos(psi),
                np.sin(psi),
            ] * distance
        elif virtual_site_key.type == "DivalentLonePair":
            distance = potential.parameters["distance"]
            theta = potential.parameters["inPlaneAngle"].m_as(unit.radian)
            local_frame_position = [
                -1.0 * np.cos(theta),
                0.0,
                np.sin(theta),
            ] * distance
        elif virtual_site_key.type == "TrivalentLonePair":
            distance = potential.parameters["distance"]
            local_frame_position = [-1.0, 0.0, 0.0] * distance

        return local_frame_position


def library_charge_from_molecule(
    molecule: "Molecule",
) -> LibraryChargeHandler.LibraryChargeType:
    """Given an OpenFF Molecule with charges, generate a corresponding LibraryChargeType"""
    if molecule.partial_charges is None:
        raise ValueError("Input molecule is missing partial charges.")

    smirks = molecule.to_smiles(mapped=True)
    charges = molecule.partial_charges

    library_charge_type = LibraryChargeHandler.LibraryChargeType(
        smirks=smirks, charge=charges
    )

    return library_charge_type


SMIRNOFF_POTENTIAL_HANDLERS = [
    SMIRNOFFBondHandler,
    SMIRNOFFConstraintHandler,
    SMIRNOFFAngleHandler,
    SMIRNOFFProperTorsionHandler,
    SMIRNOFFImproperTorsionHandler,
    SMIRNOFFvdWHandler,
    SMIRNOFFElectrostaticsHandler,
]
