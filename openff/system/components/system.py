from typing import Dict, Optional, Union

import numpy as np
from openforcefield.topology.topology import Topology
from pydantic import BaseModel, validator
from simtk import unit as omm_unit

from openff.system.components.potentials import PotentialHandler
from openff.system.interop.parmed import to_parmed


class System(BaseModel):
    """
    A fake system meant only to demonstrate how `PotentialHandler`s are
    meant to be structured

    """

    handlers: Dict[str, PotentialHandler] = dict()
    topology: Optional[Topology] = None
    box: Optional[Union[omm_unit.Quantity, np.ndarray]] = None
    positions: Optional[Union[omm_unit.Quantity, np.ndarray]] = None

    class Config:
        arbitrary_types_allowed = True

    @validator("box")
    def validate_box(cls, val):
        if val is None:
            return val
        elif type(val) == omm_unit.Quantity:
            val = val / omm_unit.nanometer
        elif type(val) == np.ndarray:
            pass
        if val.shape == (3, 3):
            return val
        elif val.shape == (3,):
            val = val * np.eye(3)
            return val
        else:
            raise ValueError  # InvalidBoxError


System.to_parmed = to_parmed
