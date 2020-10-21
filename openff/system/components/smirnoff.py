from typing import Dict, Set

from openforcefield.topology.topology import Topology
from openforcefield.typing.engines.smirnoff.parameters import BondHandler

from openff.system.components.potentials import Potential, PotentialHandler
from openff.system.utils import simtk_to_pint


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
            # ParameterHandler.get_parameter returns a list, although this
            # should only ever be length 1
            parameter_type = parameter_handler.get_parameter({"smirks": smirks})[0]
            potential = Potential(
                parameters={
                    "k": simtk_to_pint(parameter_type.k),
                    "length": simtk_to_pint(parameter_type.length),
                },
            )
            self.potentials[smirks] = potential
