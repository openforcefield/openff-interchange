from typing import Dict

import numpy as np
from pydantic import BaseModel, validator, root_validator
import pint
from qcelemental.models.types import Array
from simtk.unit import Quantity as SimTKQuantity

from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.topology import Topology

from .typing.smirnoff import build_smirnoff_map, build_smirnoff_collection
from .collections import PotentialCollection
from .utils import simtk_to_pint

u = pint.UnitRegistry()


class System(BaseModel):
    """The OpenFF System object."""

    topology: Topology
    forcefield: ForceField = None
    potential_collection: PotentialCollection = None
    potential_map: Dict = None
    # These Array (numpy-drived qcel objects) probably should be custom pint.Quantity-derived objects
    positions: Array[float]
    box: Array[float]
    slots_map: Dict = None

    @root_validator
    def validate_forcefield_data(cls, values):
        # TODO: Replace this messy logic with something cleaner
        if values['forcefield'] is None:
            if values['potential_collection'] is None or values['potential_map'] is None:
                raise TypeError('not given an ff, need collection & map')
        if values['forcefield'] is not None:
            if values['potential_collection'] is not None and values['potential_map'] is not None:
                raise TypeError('ff redundantly specified, will not be used')
            # TODO: Let other typing engines drop in here
            values['potential_map'] = build_smirnoff_map(forcefield=values['forcefield'], topology=values['topology'])
            values['potential_collection'] = build_smirnoff_collection(forcefield=values['forcefield'])
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

    # TODO: I needed to set pre=True to get this to override the Array type. This is bad
    # and instead this attribute should be handled by a custom class that deals with
    # all of the complexity (NumPy/simtk.unit.Quantity/pint.Quantity) and spits out
    # a single thing that plays nicely with things
    @validator("positions", "box", pre=True)
    def validate_in_space(cls, val):
        if isinstance(val, SimTKQuantity):
            return simtk_to_pint(val)
        elif isinstance(val, np.ndarray):
            return val * u.nm
        elif isinstance(val, Array):
            return val
        else:
            raise TypeError

    class Config:
        arbitrary_types_allowed = True

    # TODO: Make these two functions a drop-in for different typing engines?
    def run_typing(self, forcefield, topology):
        return build_smirnoff_map(forcefield=forcefield, topology=topology)

    def get_potential_collection(self, forcefield):
        return build_smirnoff_collection(forcefield=forcefield)

    def to_file(self):
        raise NotImplementedError()

    def from_file(self):
        raise NotImplementedError()

    def to_parmed(self):
        raise NotImplementedError()

    def to_openmm(self):
        raise NotImplementedError()
