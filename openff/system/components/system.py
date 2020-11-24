from typing import Dict, Optional

import numpy as np
from openforcefield.topology.topology import Topology
from pydantic import BaseModel, validator

from openff.system.components.potentials import PotentialHandler
from openff.system.interop.parmed import to_parmed
from openff.system.types import LengthArray


class System(BaseModel):
    """
    A fake system meant only to demonstrate how `PotentialHandler`s are
    meant to be structured

    """

    handlers: Dict[str, PotentialHandler] = dict()
    topology: Optional[Topology] = None
    box: LengthArray = None
    positions: LengthArray = None

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    @validator("box")
    def validate_box(cls, val):
        if val is None:
            return val
        elif hasattr(val, "shape"):
            if val.shape == (3, 3):
                return val
            elif val.shape == (3,):
                return val * np.eye(3)
        else:
            raise ValueError


System.to_parmed = to_parmed
