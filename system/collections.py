from typing import Dict

from pydantic import BaseModel

from .potential import ParametrizedAnalyticalPotential as Potential


class PotentialHandler(BaseModel):

    name: str
    potentials: Dict[str, Potential] = dict()

    def __getitem__(self, key):
        return self.potentials[key]


class PotentialCollection(BaseModel):

    handlers: Dict[str, PotentialHandler] = dict()

    def __getitem__(self, handler_name):
        return self.handlers[handler_name]
