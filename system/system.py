from typing import Union, Iterable, Dict

from pydantic import BaseModel, validator
import pint

from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.topology import Topology

from .typing.smirnoff import build_smirnoff_map, SMIRNOFFCollection
from .collections import PotentialCollection


u = pint.UnitRegistry()


class System(BaseModel):
    """The OpenFF System object."""

    topology: Topology
    potential_collection: Union[PotentialCollection, ForceField]
    positions: Iterable = None
    box: Iterable = None
    slots_map: Dict = None

    @validator("potential_collection")
    def validate_potential_collection(cls, val):
        if isinstance(val, ForceField):
            return SMIRNOFFCollection.from_toolkit_forcefield(val)
        elif isinstance(val, PotentialCollection):
            return val
        else:
            raise TypeError

    @validator("topology")
    def validate_topology(cls, val):
        if isinstance(val, Topology):
            return val
        else:
            raise TypeError

    @validator("*")
    def dummy_validator(cls, val):
        return val

    class Config:
        arbitrary_types_allowed = True

    def run_typing(self, toolkit_forcefield, toolkit_topology):
        """Just store the slots map"""
        self.slots_map = build_smirnoff_map(
            forcefield=toolkit_forcefield,
            topology=toolkit_topology
        )

    def to_file(self):
        raise NotImplementedError()

    def from_file(self):
        raise NotImplementedError()

    def to_parmed(self):
        raise NotImplementedError()

    def to_openmm(self):
        raise NotImplementedError()
