from typing import Set

import numpy as np
from pydantic import BaseModel
import pint

from openforcefield.typing.engines.smirnoff import ForceField as ToolkitForceField
from openforcefield.topology import Topology as ToolkitTopology

from .potential import ParametrizedAnalyticalPotential as Potential


u = pint.UnitRegistry()


class Topology:

    def __init__(self, toolkit_topology):
        self._atoms = [Atom(atom.atomic_number) for atom in toolkit_topology.topology_atoms]


class ForceField:

    types: Set[Potential]


class Atom:

    def __init__(self, atomic_number):
        self._atomic_number = atomic_number

    @property
    def atomic_number(self):
        return self._atomic_number


class System(BaseModel):
    """The OpenFF System object."""

    toolkit_forcefield: ToolkitForceField = None
    toolkit_topology: ToolkitTopology = None
    topology: Topology = None
    forcefield: ForceField = None

    def create_system_from_toolkit_stuff(self):
        """Construct a System from provided ForceField and Topology."""
        self.box = [10, 10, 10] * u.nm

        self.positions = np.random.random((self.topology.n_topology_atoms, 3)) * u.nm


    class Config:
        arbitrary_types_allowed = True

    def to_file(self):
        raise NotImplementedError()

    def from_file(self):
        raise NotImplementedError()
