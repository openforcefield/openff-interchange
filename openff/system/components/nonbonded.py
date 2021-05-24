from typing import Dict

from typing_extensions import Literal

from openff.system.components.potentials import Potential, PotentialHandler
from openff.system.models import PotentialKey, TopologyKey


class BuckinghamvdWHandler(PotentialHandler):
    type: Literal["Buckingham-6"] = "Buckingham-6"
    expression: Literal["a*exp(-b*r)-c*r**-6"] = "a*exp(-b*r)-c*r**-6"
    mixing_rule: Literal["buckingham"] = "buckingham"
    method: str = "cutoff"
    cutoff: float = 9.0
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()
    scale_13: float = 0.0
    scale_14: float = 0.5
    scale_15: float = 1.0
