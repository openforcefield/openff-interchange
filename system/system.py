from typing import Iterable, Dict

from pydantic import BaseModel, validator, root_validator
import pint

from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.topology import Topology

from .typing.smirnoff import build_smirnoff_map
from .collections import PotentialCollection


u = pint.UnitRegistry()


class System(BaseModel):
    """The OpenFF System object."""

    topology: Topology
    forcefield: ForceField = None
    potential_collection: PotentialCollection = None
    potential_map: Dict = None
    positions: Iterable = None
    box: Iterable = None
    slots_map: Dict = None

    @root_validator
    def validate_forcefield_data(cls, values):
        # TODO: Replace this messy logic with something cleaner
        if values['forcefield'] is None:
            if values['potential_collection'] is None or values['potential_map'] is None:
                raise TypeError('not given an ff, need collection & map')
        if values['forcefield'] is not None and values['potential_collection'] is not None and values['potential_map'] is not None:
            raise TypeError('ff redundantly specified')
        return values

    # TODO: These valiators pretty much don't do anything now
    @validator("forcefield")
    def validate_forcefield(cls, val):
        if val is None:
            return val
        if isinstance(val, ForceField):
            return val
        else:
            raise TypeError

    @validator("topology")
    def validate_topology(cls, val):
        if isinstance(val, Topology):
            return val
        else:
            raise TypeError

    class Config:
        arbitrary_types_allowed = True

    def run_typing(self, forcefield, topology):
        """Just store the slots map"""
        self.slots_map = build_smirnoff_map(
            forcefield=forcefield,
            topology=topology,
        )

    def to_file(self):
        raise NotImplementedError()

    def from_file(self):
        raise NotImplementedError()

    def to_parmed(self):
        raise NotImplementedError()

    def to_openmm(self):
        raise NotImplementedError()
