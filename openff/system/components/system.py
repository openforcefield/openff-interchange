from typing import Dict, Optional

import numpy as np
from openforcefield.topology.topology import Topology
from pydantic import BaseModel, validator
from simtk import unit as omm_unit

from openff.system.components.potentials import PotentialHandler
from openff.system.interop.parmed import to_parmed
from openff.system.types import UnitArray
from openff.system.utils import simtk_to_pint


class System(BaseModel):
    """
    A fake system meant only to demonstrate how `PotentialHandler`s are
    meant to be structured

    """

    handlers: Dict[str, PotentialHandler] = dict()
    topology: Optional[Topology] = None
    box: Optional[UnitArray] = None
    positions: Optional[UnitArray] = None

    class Config:
        arbitrary_types_allowed = True

    @validator("box")
    def validate_box(cls, val):
        if val is None:
            return val
        if val.shape == (3, 3):
            pass
        elif val.shape == (3,):
            val *= np.eye(3)
        else:
            raise ValueError  # InvalidBoxError
        if type(val) == omm_unit.Quantity:
            val = simtk_to_pint(val)
            return val


System.to_parmed = to_parmed
