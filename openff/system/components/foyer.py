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
