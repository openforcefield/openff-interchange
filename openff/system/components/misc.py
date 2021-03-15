from typing import Dict, Set

from openff.system.components.potentials import Potential, PotentialHandler
from openff.system.models import PotentialKey, TopologyKey


class BuckinghamvdWHandler(PotentialHandler):
    name = "Buckingham-6"
    expression = "A * exp(-B *r) - C * r ** -6"
    independent_variables: Set[str] = {"r"}
    method: str = "Cutoff"
    cutoff: float = 9.0
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()
    scale_13: float = 0.0
    scale_14: float = 0.5
    scale_15: float = 1.0
