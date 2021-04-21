from typing import Dict, Set

from openff.toolkit.typing.engines.smirnoff.forcefield import ForceField
from openff.toolkit.typing.engines.smirnoff.parameters import (
    AngleHandler,
    BondHandler,
    ChargeIncrementModelHandler,
    ConstraintHandler,
    ImproperTorsionHandler,
    LibraryChargeHandler,
    ProperTorsionHandler,
    vdWHandler,
)
from openff.units import unit
from pydantic import validator
from simtk import unit as omm_unit

from openff.system.components.misc import OFFBioTop
from openff.system.components.potentials import Potential, PotentialHandler
from openff.system.exceptions import UnsupportedCutoffMethodError
from openff.system.models import DefaultModel, PotentialKey, TopologyKey
from openff.system.types import FloatQuantity
from openff.system.utils import get_partial_charges_from_openmm_system

kcal_mol = omm_unit.kilocalorie_per_mole
kcal_mol_angstroms = kcal_mol / omm_unit.angstrom ** 2
kcal_mol_radians = kcal_mol / omm_unit.radian ** 2


class SMIRNOFFBondHandler(PotentialHandler):

    name: str = "Bonds"
    expression: str = "1/2 * k * (r - length) ** 2"
    independent_variables: Set[str] = {"r"}
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()

    def store_matches(
        self, parameter_handler: BondHandler, topology: OFFBioTop
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
        if self.slot_map:
            self.slot_map = dict()
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            topology_key = TopologyKey(atom_indices=key)
            potential_key = PotentialKey(id=val.parameter_type.smirks)
            self.slot_map[topology_key] = potential_key

    def store_potentials(self, parameter_handler: BondHandler) -> None:
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


class SMIRNOFFConstraintHandler(PotentialHandler):

    name: str = "Constraints"
    expression: str = ""
    independent_variables: Set[str] = {""}
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    constraints: Dict[
        PotentialKey, bool
    ] = dict()  # should this be named potentials for consistency?

    def store_matches(
        self,
        parameter_handler: ConstraintHandler,
        topology: OFFBioTop,
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
        if self.slot_map:
            self.slot_map = dict()
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            topology_key = TopologyKey(atom_indices=key)
            potential_key = PotentialKey(id=val.parameter_type.smirks)
            self.slot_map[topology_key] = potential_key

    def store_constraints(
        self,
        parameter_handler: ConstraintHandler,
        bond_handler: SMIRNOFFBondHandler = None,
    ) -> None:
        """
        Populate self.constraints with key-val pairs of unique potential
        identifiers and their associated Potential objects

        TODO: Raname to store_potentials potentials for consistency?

        """
        if self.constraints:
            self.constraints = dict()
        for top_key, pot_key in self.slot_map.items():
            smirks = pot_key.id
            parameter_type = parameter_handler.get_parameter({"smirks": smirks})[0]
            if parameter_type.distance:
                distance = parameter_type.distance
            else:
                if not bond_handler:
                    from openff.system.exceptions import MissingParametersError

                    raise MissingParametersError(
                        f"Constraint with SMIRKS pattern {smirks} found with no distance "
                        "specified, and no corresponding bond parameters were found. The distance "
                        "of this constraint is not specified."
                    )
                # Look up by atom indices because constraint and bond SMIRKS may not match
                bond_key = bond_handler.slot_map[top_key]
                bond_parameter = bond_handler.potentials[bond_key].parameters
                distance = bond_parameter["length"]
            potential = Potential(
                parameters={
                    "distance": distance,
                }
            )
            self.constraints[pot_key] = potential  # type: ignore[assignment]


class SMIRNOFFAngleHandler(PotentialHandler):

    name: str = "Angles"
    expression: str = "1/2 * k * (angle - theta)"
    independent_variables: Set[str] = {"theta"}
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()

    def store_matches(
        self,
        parameter_handler: AngleHandler,
        topology: OFFBioTop,
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            topology_key = TopologyKey(atom_indices=key)
            potential_key = PotentialKey(id=val.parameter_type.smirks)
            self.slot_map[topology_key] = potential_key

    def store_potentials(self, parameter_handler: AngleHandler) -> None:
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


class SMIRNOFFProperTorsionHandler(PotentialHandler):

    name: str = "ProperTorsions"
    expression: str = "k*(1+cos(periodicity*theta-phase))"
    independent_variables: Set[str] = {"theta"}
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()

    def store_matches(
        self, parameter_handler: ProperTorsionHandler, topology: OFFBioTop
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            n_terms = len(val.parameter_type.k)
            for n in range(n_terms):
                smirks = val.parameter_type.smirks
                topology_key = TopologyKey(atom_indices=key, mult=n)
                potential_key = PotentialKey(id=smirks, mult=n)
                self.slot_map[topology_key] = potential_key

    def store_potentials(self, parameter_handler: ProperTorsionHandler) -> None:
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


class SMIRNOFFImproperTorsionHandler(PotentialHandler):

    name: str = "ImproperTorsions"
    expression: str = "k*(1+cos(periodicity*theta-phase))"
    independent_variables: Set[str] = {"theta"}
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()

    def store_matches(
        self, parameter_handler: ImproperTorsionHandler, topology: OFFBioTop
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
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
                topology_key = TopologyKey(atom_indices=key, mult=n)
                potential_key = PotentialKey(id=smirks, mult=n)
                self.slot_map[topology_key] = potential_key

    def store_potentials(self, parameter_handler: ImproperTorsionHandler) -> None:
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


class SMIRNOFFvdWHandler(PotentialHandler):

    name: str = "vdW"
    expression: str = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"
    independent_variables: Set[str] = {"r"}
    method: str = "cutoff"
    cutoff: FloatQuantity["angstrom"] = 9.0 * unit.angstrom  # type: ignore
    switch_width: FloatQuantity["angstrom"] = 1.0 * unit.angstrom  # type: ignore
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()
    scale_13: float = 0.0
    scale_14: float = 0.5
    scale_15: float = 1.0

    @validator("method")
    def validate_method(cls, v):
        v_ = v.lower().replace("-", "")
        if v_ != "cutoff":
            raise UnsupportedCutoffMethodError(f"vdW cutoff method {v} not supported")
        return v

    def store_matches(
        self,
        parameter_handler: vdWHandler,
        topology: OFFBioTop,
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            topology_key = TopologyKey(atom_indices=key)
            potential_key = PotentialKey(id=val.parameter_type.smirks)
            self.slot_map[topology_key] = potential_key

    def store_potentials(self, parameter_handler: vdWHandler) -> None:
        """
        Populate self.potentials with key-val pairs of unique potential
        identifiers and their associated Potential objects

        """
        self.method = parameter_handler.method
        self.cutoff = parameter_handler.cutoff.value_in_unit(omm_unit.angstrom)

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


class SMIRNOFFElectrostaticsMetadataMixin(DefaultModel):

    name: str = "Electrostatics"
    expression: str = "coul"
    method: str = "PME"
    cutoff: FloatQuantity["angstrom"] = 9.0  # type: ignore
    independent_variables: Set[str] = {"r"}
    charge_map: Dict[TopologyKey, float] = dict()
    scale_13: float = 0.0
    scale_14: float = 0.8333333333
    scale_15: float = 1.0

    def store_charges(
        self,
        forcefield: ForceField,
        topology: OFFBioTop,
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
        self.method = forcefield["Electrostatics"].method

        partial_charges = get_partial_charges_from_openmm_system(
            forcefield.create_openmm_system(topology=topology)
        )

        for i, charge in enumerate(partial_charges):
            topology_key = TopologyKey(atom_indices=(i,))
            self.charge_map[topology_key] = charge * unit.elementary_charge


class SMIRNOFFLibraryChargeHandler(  # type: ignore[misc]
    SMIRNOFFElectrostaticsMetadataMixin,
    PotentialHandler,
):

    name: str = "LibraryCharges"
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()

    def store_matches(
        self,
        parameter_handler: LibraryChargeHandler,
        topology: OFFBioTop,
    ) -> None:
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            top_key = TopologyKey(atom_indices=key)
            pot_key = PotentialKey(id=val.parameter_type.smirks)
            self.slot_map[top_key] = pot_key

    def store_potentials(self, parameter_handler: LibraryChargeHandler) -> None:
        if self.potentials:
            self.potentials = dict()
        for potential_key in self.slot_map.values():
            smirks = potential_key.id
            parameter_type = parameter_handler.get_parameter({"smirks": smirks})[0]
            charges_unitless = [val._value for val in parameter_type.charge]
            potential = Potential(
                parameters={"charges": charges_unitless * unit.elementary_charge},
            )
            self.potentials[potential_key] = potential


class SMIRNOFFChargeIncrementHandler(  # type: ignore[misc]
    SMIRNOFFElectrostaticsMetadataMixin,
    PotentialHandler,
):

    name: str = "ChargeIncrements"
    partial_charge_method: str = "AM1-Mulliken"
    potentials: Dict[PotentialKey, Potential] = dict()

    def store_matches(
        self,
        parameter_handler: ChargeIncrementModelHandler,
        topology: OFFBioTop,
    ) -> None:
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            top_key = TopologyKey(atom_indices=key)
            pot_key = PotentialKey(id=val.parameter_type.smirks)
            self.slot_map[top_key] = pot_key

    def store_potentials(self, parameter_handler: ChargeIncrementModelHandler) -> None:
        if self.potentials:
            self.potentials = dict()
        for potential_key in self.slot_map.values():
            smirks = potential_key.id
            parameter_type = parameter_handler.get_parameter({"smirks": smirks})[0]
            charges_unitless = [val._value for val in parameter_type.charge_increment]
            potential = Potential(
                parameters={
                    "charge_increments": charges_unitless * unit.elementary_charge
                },
            )
            self.potentials[potential_key] = potential


class ElectrostaticsMetaHandler(SMIRNOFFElectrostaticsMetadataMixin):

    name: str = "Electrostatics"
    charges: Dict = dict()  # type
    cache: Dict = dict()  # Dict[str: Dict[str, FloatQuantity["elementary_charge"]]]

    def cache_charges(self, partial_charge_method: str, topology: OFFBioTop):

        charges: Dict[TopologyKey, FloatQuantity] = dict()

        for ref_mol in topology.reference_molecules:
            ref_mol.assign_partial_charges(partial_charge_method=partial_charge_method)

            for top_mol in topology._reference_molecule_to_topology_molecules[ref_mol]:
                for topology_particle in top_mol.atoms:
                    ref_mol_particle_index = (
                        topology_particle.atom.molecule_particle_index
                    )
                    topology_particle_index = topology_particle.topology_particle_index
                    partial_charge = ref_mol._partial_charges[ref_mol_particle_index]
                    partial_charge = partial_charge.value_in_unit(
                        omm_unit.elementary_charge
                    )
                    partial_charge = partial_charge * unit.elementary_charge
                    top_key = TopologyKey(atom_indices=(topology_particle_index,))
                    charges[top_key] = partial_charge

        self.cache[partial_charge_method] = charges

    def apply_charge_increments(
        self, charge_increments: SMIRNOFFChargeIncrementHandler
    ):
        for top_key, pot_key in charge_increments.slot_map.items():
            ids = top_key.atom_indices
            charges = charge_increments.potentials[pot_key].parameters[
                "charge_increments"
            ]
            for i, id_ in enumerate(ids):
                atom_key = TopologyKey(atom_indices=(id_,))
                self.charges[atom_key] += charges[i]

    def apply_library_charges(self, library_charges: SMIRNOFFLibraryChargeHandler):
        for top_key, pot_key in library_charges.slot_map.items():
            ids = top_key.atom_indices
            charges = library_charges.potentials[pot_key].parameters["charges"]
            # Need to ensure this iterator follows ordering in force field
            for i, id_ in enumerate(ids):
                atom_key = TopologyKey(atom_indices=(id_,))
                self.charges[atom_key] = charges[i]
