from typing import Dict, Set

from openforcefield.topology.topology import Topology
from openforcefield.typing.engines.smirnoff.forcefield import ForceField
from openforcefield.typing.engines.smirnoff.parameters import (
    AngleHandler,
    BondHandler,
    ConstraintHandler,
    ProperTorsionHandler,
    vdWHandler,
)
from pydantic import BaseModel
from simtk import unit as omm_unit

from openff.system.components.potentials import Potential, PotentialHandler
from openff.system.utils import get_partial_charges_from_openmm_system

kcal_mol = omm_unit.kilocalorie_per_mole
kcal_mol_angstroms = kcal_mol / omm_unit.angstrom ** 2
kcal_mol_radians = kcal_mol / omm_unit.radian ** 2


class SMIRNOFFConstraintHandler(PotentialHandler):

    name: str = "Constraints"
    expression: str = ""
    independent_variables: Set[str] = {""}
    slot_map: Dict[str, str] = dict()
    constraints: Dict[
        str, bool
    ] = dict()  # should this be named potentials for consistency?

    def store_matches(
        self, parameter_handler: ConstraintHandler, topology: Topology
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
        if self.slot_map:
            self.slot_map = dict()
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            key = str(key)
            self.slot_map[key] = val.parameter_type.smirks

    def store_constraints(self, parameter_handler: ConstraintHandler) -> None:
        """
        Populate self.constraints with key-val pairs of unique potential
        identifiers and their associated Potential objects

        TODO: Raname to store_potentials potentials for consistency?

        """
        if self.constraints:
            self.constraints = dict()
        for smirks in self.slot_map.values():
            # Simply store _if_ this slot is to be constrained;
            # let the details be dealt with by the interoperability layer
            self.constraints[smirks] = True


class SMIRNOFFBondHandler(PotentialHandler):

    name: str = "Bonds"
    expression: str = "1/2 * k * (r - length) ** 2"
    independent_variables: Set[str] = {"r"}
    slot_map: Dict[str, str] = dict()
    potentials: Dict[str, Potential] = dict()

    def store_matches(self, parameter_handler: BondHandler, topology: Topology) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
        if self.slot_map:
            self.slot_map = dict()
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            key = str(key)
            self.slot_map[key] = val.parameter_type.smirks

    def store_potentials(self, parameter_handler: BondHandler) -> None:
        """
        Populate self.potentials with key-val pairs of unique potential
        identifiers and their associated Potential objects

        """
        if self.potentials:
            self.potentials = dict()
        for smirks in self.slot_map.values():
            parameter_type = parameter_handler.get_parameter({"smirks": smirks})[0]
            potential = Potential(
                parameters={
                    "k": parameter_type.k / kcal_mol_angstroms,
                    "length": parameter_type.length / omm_unit.angstrom,
                },
            )
            self.potentials[smirks] = potential


class SMIRNOFFAngleHandler(PotentialHandler):

    name: str = "Angles"
    expression: str = "1/2 * k * (angle - theta)"
    independent_variables: Set[str] = {"theta"}
    slot_map: Dict[str, str] = dict()
    potentials: Dict[str, Potential] = dict()

    def store_matches(
        self, parameter_handler: AngleHandler, topology: Topology
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            key = str(key)
            self.slot_map[key] = val.parameter_type.smirks

    def store_potentials(self, parameter_handler: AngleHandler) -> None:
        """
        Populate self.potentials with key-val pairs of unique potential
        identifiers and their associated Potential objects

        """
        for smirks in self.slot_map.values():
            # ParameterHandler.get_parameter returns a list, although this
            # should only ever be length 1
            parameter_type = parameter_handler.get_parameter({"smirks": smirks})[0]
            potential = Potential(
                parameters={
                    "k": parameter_type.k / kcal_mol_radians,
                    "angle": parameter_type.angle / omm_unit.degree,
                },
            )
            self.potentials[smirks] = potential


class SMIRNOFFProperTorsionHandler(PotentialHandler):

    name: str = "ProperTorsions"
    expression: str = "k*(1+cos(periodicity*theta-phase))"
    independent_variables: Set[str] = {"theta"}
    slot_map: Dict[str, str] = dict()
    potentials: Dict[str, Potential] = dict()

    def store_matches(
        self, parameter_handler: ProperTorsionHandler, topology: Topology
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            n_terms = len(val.parameter_type.k)
            for n in range(n_terms):
                # This (later) assumes that `_` is disallowed in SMIRKS ...
                identifier = str(key) + f"_{n}"
                self.slot_map[identifier] = val.parameter_type.smirks + f"_{n}"

    def store_potentials(self, parameter_handler: ProperTorsionHandler) -> None:
        """
        Populate self.potentials with key-val pairs of unique potential
        identifiers and their associated Potential objects

        """
        for key in self.slot_map.values():
            # ParameterHandler.get_parameter returns a list, although this
            # should only ever be length 1
            smirks = key.split("_")[0]
            parameter_type = parameter_handler.get_parameter({"smirks": smirks})[0]
            n_terms = len(parameter_type.k)
            for n in range(n_terms):
                identifier = key
                potential = Potential(
                    parameters={
                        "k": parameter_type.k[n] / kcal_mol,
                        "periodicity": parameter_type.periodicity[n],
                        "phase": parameter_type.phase[n] / omm_unit.degree,
                    },
                )
                self.potentials[identifier] = potential


class SMIRNOFFvdWHandler(PotentialHandler):

    name: str = "vdW"
    expression: str = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"
    independent_variables: Set[str] = {"r"}
    method: str = "Cutoff"
    cutoff: float = 9.0
    slot_map: Dict[str, str] = dict()
    potentials: Dict[str, Potential] = dict()
    scale_13: float = 0.0
    scale_14: float = 0.5
    scale_15: float = 1.0

    def store_matches(
        self,
        parameter_handler: vdWHandler,
        topology: Topology,
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            key = str(key)
            self.slot_map[key] = val.parameter_type.smirks

    def store_potentials(self, parameter_handler: vdWHandler) -> None:
        """
        Populate self.potentials with key-val pairs of unique potential
        identifiers and their associated Potential objects

        """
        self.method = parameter_handler.method
        self.cutoff = parameter_handler.cutoff / omm_unit.angstrom

        for smirks in self.slot_map.values():
            parameter_type = parameter_handler.get_parameter({"smirks": smirks})[0]
            try:
                potential = Potential(
                    parameters={
                        "sigma": parameter_type.sigma / omm_unit.angstrom,
                        "epsilon": parameter_type.epsilon / kcal_mol,
                    },
                )
            except AttributeError:
                # Handle rmin_half pending https://github.com/openforcefield/openforcefield/pull/750
                potential = Potential(
                    parameters={
                        "sigma": parameter_type.sigma / omm_unit.angstrom,
                        "epsilon": parameter_type.epsilon / kcal_mol,
                    },
                )
            self.potentials[smirks] = potential


class SMIRNOFFElectrostaticsHandler(BaseModel):

    name: str = "Electrostatics"
    expression: str = "coul"
    independent_variables: Set[str] = {"r"}
    charge_map: Dict[str, float] = dict()
    scale_13: float = 0.0
    scale_14: float = 0.8333333333
    scale_15: float = 1.0

    def store_charges(
        self,
        forcefield: ForceField,
        topology: Topology,
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
        partial_charges = get_partial_charges_from_openmm_system(
            forcefield.create_openmm_system(topology=topology)
        )  # / omm_unit.elementary_charge

        for i, charge in enumerate(partial_charges):
            self.charge_map[str((i,))] = charge

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True


SUPPORTED_HANDLER_MAPPING = {
    "Constriants": SMIRNOFFConstraintHandler,
    "Bonds": SMIRNOFFBondHandler,
    "Angles": SMIRNOFFAngleHandler,
    "vdW": SMIRNOFFvdWHandler,
}
