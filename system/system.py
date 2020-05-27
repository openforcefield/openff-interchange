from typing import Iterable

from pydantic import BaseModel, validator

from openforcefield.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField


class System(BaseModel):
    """The OpenFF System object."""

    topology: Topology = None
    forcefield: ForceField = None
    positions: Iterable = None
    box: Iterable = None

    @validator("*")
    def dummy_validator(cls, val):
        return val

    class Config:
        arbitrary_types_allowed = True

    def to_file(self):
        raise NotImplementedError()

    def from_file(self):
        raise NotImplementedError()

    def to_parmed(self):
        raise NotImplementedError()

    def to_openmm(self):
        raise NotImplementedError()
