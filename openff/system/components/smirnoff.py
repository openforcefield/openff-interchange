from typing import Dict, Set

from openforcefield.topology.topology import Topology
from openforcefield.typing.engines.smirnoff.forcefield import ForceField
from openforcefield.typing.engines.smirnoff.parameters import (
    AngleHandler,
    BondHandler,
    ProperTorsionHandler,
    vdWHandler,
)
from pydantic import BaseModel

from openff.system import unit
from openff.system.components.potentials import Potential, PotentialHandler
from openff.system.utils import get_partial_charges_from_openmm_system, simtk_to_pint


class SMIRNOFFBondHandler(PotentialHandler):

    name: str = "Bonds"
    expression: str = "1/2 * k * (r - length) ** 2"
    independent_variables: Set[str] = {"r"}
    slot_map: Dict[tuple, str] = dict()
    potentials: Dict[str, Potential] = dict()

    def store_matches(self, parameter_handler: BondHandler, topology: Topology) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            self.slot_map[key] = val.parameter_type.smirks

    def store_potentials(self, parameter_handler: BondHandler) -> None:
        """
        Populate self.potentials with key-val pairs of unique potential
        identifiers and their associated Potential objects

        """
        for smirks in self.slot_map.values():
            parameter_type = parameter_handler.get_parameter({"smirks": smirks})[0]
            potential = Potential(
                parameters={
                    "k": simtk_to_pint(parameter_type.k),
                    "length": simtk_to_pint(parameter_type.length),
                },
            )
            self.potentials[smirks] = potential


class SMIRNOFFAngleHandler(PotentialHandler):

    name: str = "Angles"
    expression: str = "1/2 * k * (angle - theta)"
    independent_variables: Set[str] = {"theta"}
    slot_map: Dict[tuple, str] = dict()
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
                    "k": simtk_to_pint(parameter_type.k),
                    "angle": simtk_to_pint(parameter_type.angle),
                },
            )
            self.potentials[smirks] = potential


class SMIRNOFFProperTorsionHandler(PotentialHandler):

    name: str = "ProperTorsions"
    expression: str = "k*(1+cos(periodicity*theta-phase))"
    independent_variables: Set[str] = {"theta"}
    slot_map: Dict[tuple, str] = dict()
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
            self.slot_map[key] = val.parameter_type.smirks

    def store_potentials(self, parameter_handler: ProperTorsionHandler) -> None:
        """
        Populate self.potentials with key-val pairs of unique potential
        identifiers and their associated Potential objects

        """
        for smirks in self.slot_map.values():
            # ParameterHandler.get_parameter returns a list, although this
            # should only ever be length 1
            parameter_type = parameter_handler.get_parameter({"smirks": smirks})[0]
            n_terms = len(parameter_type.k)
            potential = Potential(
                parameters={
                    "k": [simtk_to_pint(val) for val in parameter_type.k],
                    "periodicity": parameter_type.periodicity,
                    "phase": [simtk_to_pint(val) for val in parameter_type.phase],
                    "n_terms": n_terms,
                },
            )
            self.potentials[smirks] = potential


class SMIRNOFFvdWHandler(PotentialHandler):

    name: str = "vdW"
    expression: str = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"
    independent_variables: Set[str] = {"r"}
    slot_map: Dict[tuple, str] = dict()
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
            self.slot_map[key] = val.parameter_type.smirks

    def store_potentials(self, parameter_handler: vdWHandler) -> None:
        """
        Populate self.potentials with key-val pairs of unique potential
        identifiers and their associated Potential objects

        """
        for smirks in self.slot_map.values():
            parameter_type = parameter_handler.get_parameter({"smirks": smirks})[0]
            try:
                potential = Potential(
                    parameters={
                        "sigma": simtk_to_pint(parameter_type.sigma),
                        "epsilon": simtk_to_pint(parameter_type.epsilon),
                    },
                )
            except AttributeError:
                # Handle rmin_half pending https://github.com/openforcefield/openforcefield/pull/750
                potential = Potential(
                    parameters={
                        "sigma": simtk_to_pint(parameter_type.rmin_half / 2 ** (1 / 6)),
                        "epsilon": simtk_to_pint(parameter_type.epsilon),
                    },
                )
            self.potentials[smirks] = potential


class SMIRNOFFElectrostaticsHandler(BaseModel):

    name: str = "Electrostatics"
    expression: str = "coul"
    independent_variables: Set[str] = {"r"}
    charge_map: Dict[tuple, unit.Quantity] = dict()
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
        )

        for i, charge in enumerate(partial_charges):
            self.charge_map[(i,)] = partial_charges[i]

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True


SUPPORTED_HANDLER_MAPPING = {
    "Bonds": SMIRNOFFBondHandler,
    "Angles": SMIRNOFFAngleHandler,
    "vdW": SMIRNOFFvdWHandler,
}
