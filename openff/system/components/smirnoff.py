import abc
import copy
import functools
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ChargeIncrementModelHandler,
    ConstraintHandler,
    ElectrostaticsHandler,
    LibraryChargeHandler,
    ParameterHandler,
    vdWHandler,
)
from openff.units import unit
from pydantic import Field
from simtk import unit as omm_unit
from typing_extensions import Literal

from openff.system.components.potentials import Potential, PotentialHandler
from openff.system.models import PotentialKey, TopologyKey
from openff.system.types import FloatQuantity

kcal_mol = omm_unit.kilocalorie_per_mole
kcal_mol_angstroms = kcal_mol / omm_unit.angstrom ** 2
kcal_mol_radians = kcal_mol / omm_unit.radian ** 2

if TYPE_CHECKING:
    from openff.toolkit.topology.topology import Topology
    from openff.toolkit.typing.engines.smirnoff.parameters import (
        AngleHandler,
        BondHandler,
        ImproperTorsionHandler,
        ProperTorsionHandler,
        ToolkitAM1BCCHandler,
    )

    from openff.system.components.mdtraj import OFFBioTop

    ElectrostaticsHandlerType = Union[
        ElectrostaticsHandler,
        ChargeIncrementModelHandler,
        LibraryChargeHandler,
        ToolkitAM1BCCHandler,
    ]


T = TypeVar("T", bound="SMIRNOFFPotentialHandler")
T_ = TypeVar("T_", bound="PotentialHandler")


class SMIRNOFFPotentialHandler(PotentialHandler, abc.ABC):

    def store_matches(
        self,
        parameter_handler: ParameterHandler,
        topology: Union["Topology", "OFFBioTop"],
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
        if self.slot_map:
            # TODO: Should the slot_map always be reset, or should we be able to partially
            # update it? Also Note the duplicated code in the child classes
            self.slot_map = dict()
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            topology_key = TopologyKey(atom_indices=key)
            potential_key = PotentialKey(id=val.parameter_type.smirks)
            self.slot_map[topology_key] = potential_key

    @classmethod
    def from_toolkit(
        cls: Type[T],
        parameter_handler: T_,
        topology: "Topology",
    ) -> T:
        """
        Create a SMIRNOFFPotentialHandler from toolkit data.

        """
        handler = cls()
        handler.store_matches(parameter_handler=parameter_handler, topology=topology)
        handler.store_potentials(parameter_handler=parameter_handler)

        return handler


class SMIRNOFFBondHandler(SMIRNOFFPotentialHandler):

    type: Literal["Bonds"] = "Bonds"
    expression: Literal["k/2*(r-length)**2"] = "k/2*(r-length)**2"

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
    def from_toolkit(  # type: ignore[override]
        cls: Type[T],
        bond_handler: "BondHandler",
        topology: "Topology",
        constraint_handler: Optional[T] = None,
    ) -> Tuple[T, Optional["SMIRNOFFConstraintHandler"]]:
        """
        Create a SMIRNOFFBondHandler from toolkit data.

        """
        handler = cls(type="Bonds", expression="k/2*(r-length)**2")
        handler.store_matches(parameter_handler=bond_handler, topology=topology)
        handler.store_potentials(parameter_handler=bond_handler)

        if constraint_handler:
            constraints: Optional[
                "SMIRNOFFConstraintHandler"
            ] = SMIRNOFFConstraintHandler()
            constraints.store_constraints(  # type: ignore[union-attr]
                parameter_handler=constraint_handler,
                topology=topology,
                bond_handler=handler,
            )
        else:
            constraints = None

        return handler, constraints


class SMIRNOFFConstraintHandler(SMIRNOFFPotentialHandler):

    type: Literal["Constraints"] = "Constraints"
    expression: Literal[""] = ""
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    constraints: Dict[
        PotentialKey, bool
    ] = dict()  # should this be named potentials for consistency?

    def store_constraints(
        self,
        parameter_handler: ConstraintHandler,
        topology: "OFFBioTop",
        bond_handler: SMIRNOFFBondHandler = None,
    ) -> None:

        if self.slot_map:
            self.slot_map = dict()
        matches = parameter_handler.find_matches(topology)
        for key, match in matches.items():
            topology_key = TopologyKey(atom_indices=key)
            smirks = match.parameter_type.smirks
            distance = match.parameter_type.distance
            if distance is not None:
                # This constraint parameter is fully specified
                potential_key = PotentialKey(id=smirks)
                distance = match.parameter_type.distance
            else:
                # This constraint parameter depends on the BondHandler
                if not bond_handler:
                    from openff.system.exceptions import MissingParametersError

                    raise MissingParametersError(
                        f"Constraint with SMIRKS pattern {smirks} found with no distance "
                        "specified, and no corresponding bond parameters were found. The distance "
                        "of this constraint is not specified."
                    )
                # so use the same PotentialKey instance as the BondHandler
                potential_key = bond_handler.slot_map[topology_key]
                self.slot_map[topology_key] = potential_key
                distance = bond_handler.potentials[potential_key].parameters["length"]
            potential = Potential(
                parameters={
                    "distance": distance,
                }
            )
            self.constraints[potential_key] = potential  # type: ignore[assignment]


class SMIRNOFFAngleHandler(SMIRNOFFPotentialHandler):

    type: Literal["Angles"] = "Angles"
    expression: Literal["k/2*(theta-angle)**2"] = "k/2*(theta-angle)**2"

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
    def from_toolkit(
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
                potential_key = PotentialKey(id=smirks, mult=n)
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

    @classmethod
    def _from_toolkit(
        cls: Type[T],
        parameter_handler: "ProperTorsionHandler",
        topology: "Topology",
    ) -> T:

        handler = cls()
        handler.store_matches(parameter_handler=parameter_handler, topology=topology)
        handler.store_potentials(parameter_handler=parameter_handler)

        return handler


class SMIRNOFFImproperTorsionHandler(SMIRNOFFPotentialHandler):

    type: Literal["ImproperTorsions"] = "ImproperTorsions"
    expression: Literal[
        "k*(1+cos(periodicity*theta-phase))"
    ] = "k*(1+cos(periodicity*theta-phase))"

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
                    potential_key = PotentialKey(id=smirks, mult=n)
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

    @classmethod
    def _from_toolkit(
        cls: Type[T],
        improper_torsion_Handler: "ImproperTorsionHandler",
        topology: "Topology",
    ) -> T:

        handler = cls()
        handler.store_matches(
            parameter_handler=improper_torsion_Handler, topology=topology
        )
        handler.store_potentials(parameter_handler=improper_torsion_Handler)

        return handler


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
        1.0 * unit.angstrom,  # type: ignore
        description="The width over which the switching function is applied",
    )

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
        handler = SMIRNOFFvdWHandler(
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

    @property
    def partial_charges(self) -> Dict[TopologyKey, unit.Quantity]:
        """Returns the total partial charge on each particle in the associated system."""

        charges = defaultdict(lambda: 0.0)

        for topology_key, potential_key in self.slot_map.items():

            potential = self.potentials[potential_key]

            for parameter_key, parameter_value in potential.parameters.items():

                if parameter_key != "charge" and parameter_key != "charge_increment":
                    raise NotImplementedError()

                charge = parameter_value.to(unit.elementary_charge).magnitude
                charges[topology_key.atom_indices[0]] += charge

        return (
            numpy.array([charges[i] for i in range(len(charges))])
            * unit.elementary_charge
        )

    @classmethod
    def charge_precedence(cls) -> List[str]:
        """The order in which parameter handlers take precedence when computing the
        charges
        """
        return ["LibraryCharges", "ChargeIncrementModel", "ToolkitAM1BCC"]

    @classmethod
    @functools.lru_cache
    def _compute_partial_charges(cls, molecule: Molecule, method: str) -> unit.Quantity:

        from simtk import unit as simtk_unit

        molecule = copy.deepcopy(molecule)
        molecule.assign_partial_charges(method)

        return (
            molecule.partial_charges.value_in_unit(simtk_unit.elementary_charge)
            * unit.elementary_charge
        )

    @classmethod
    def _library_charge_to_potentials(
        cls,
        atom_indices: Tuple[int, ...],
        parameter: LibraryChargeHandler.LibraryChargeType,
    ) -> Tuple[Dict[TopologyKey, PotentialKey], Dict[PotentialKey, Potential]]:
        """Maps a matched library charge parameter to a set of potentials."""
        from simtk import unit as simtk_unit

        matches = {}
        potentials = {}

        for i, (atom_index, charge) in enumerate(zip(atom_indices, parameter.charge)):
            topology_key = TopologyKey(atom_indices=(atom_index,))
            potential_key = PotentialKey(id=parameter.smirks, mult=i)

            potential = Potential(
                parameters={
                    "charge": charge.value_in_unit(simtk_unit.elementary_charge)
                    * unit.elementary_charge
                }
            )

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
        from simtk import unit as simtk_unit

        matches = {}
        potentials = {}

        for i, atom_index in enumerate(atom_indices):
            topology_key = TopologyKey(atom_indices=(atom_index,))
            potential_key = PotentialKey(id=parameter.smirks, mult=i)

            # TODO: Handle the cases where n - 1 charge increments have been defined,
            #       maybe by implementing this in the TK?
            charge_increment = getattr(parameter, f"charge_increment{i + 1}")

            potential = Potential(
                parameters={
                    "charge_increment": charge_increment.value_in_unit(
                        simtk_unit.elementary_charge
                    )
                    * unit.elementary_charge
                }
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
        """Constructs a slot and potential map for a slot based parameter handler.
        """
        parameter_matches = parameter_handler.find_matches(
            reference_molecule.to_topology()
        )

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
        """Constructs a slot and potential map for a charge model based parameter handler.
        """

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

            potential_key = PotentialKey(id=reference_smiles, mult=i)
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

        for handler_type in cls.charge_precedence():

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
