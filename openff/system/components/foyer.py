from typing import Dict, Set

import parmed as pmd
from foyer import Forcefield

from openff.system.components.potentials import Potential, PotentialHandler


class FoyerAtomTypes(PotentialHandler):

    name: str = "Atoms"
    expression: str = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"
    independent_variables: Set[str] = {"r"}
    slot_map: Dict[int, str] = dict()
    potentials: Dict[str, Potential] = dict()

    def store_matches(
        self,
        forcefield: Forcefield,
        structure: pmd.Structure,
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
        type_map = forcefield.run_atomtyping(structure)

        for key, val in type_map.items():
            self.slot_map[key] = val["atomtype"]

    def store_potentials(self):
        """"""


class FoyerBondHandler(PotentialHandler):

    name: str = "Bonds"
    expression: str = "1/2 * k * (r - length) ** 2"
    independent_variables: Set[str] = {"r"}
    slot_map: Dict[str, str] = dict()
    potentials: Dict[str, Potential] = dict()

    def make_slot_map(self):
        pass
        # populates a dict with the structure like
        # self.slot_map = {"(0, 1)": ["opls_140", "opls_141"]}

    def lookup_key(self):
        for identifier in self.slot_map.values():
            # look up FF data from identifier (or directly consult the
            # topology, which would involve droppint the make_slot_map method)
            # store FF data into a Potential object
            self.potentials[identifier] = Potential(...)
