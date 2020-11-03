from typing import Dict

from pydantic import BaseModel

from openff.system.components.potentials import PotentialHandler


class System(BaseModel):
    """
    A fake system meant only to demonstrate how `PotentialHandler`s are
    meant to be structured

    """

    handlers: Dict[str, PotentialHandler] = dict()

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
