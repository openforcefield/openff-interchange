from typing import Iterable, Set

import numpy as np
from pydantic import BaseModel, validator
import pint

from openforcefield.typing.engines.smirnoff import ForceField as ToolkitForceField
from openforcefield.topology import Topology as ToolkitTopology
from openforcefield.topology.molecule import Atom as ToolkitAtom

from .potential import ParametrizedAnalyticalPotential as Potential
from .utils import simtk_to_pint


u = pint.UnitRegistry()


class Topology:

    def __init__(self, toolkit_topology):
        self.atoms = [Atom(atom.atomic_number) for atom in toolkit_topology.topology_atoms]

class ForceField:

    types: Set[Potential]


class Atom(ToolkitAtom):

    def __init__(self, atomic_number, atom_type=None):
        super().__init__(
            atomic_number=atomic_number,
            formal_charge=0,
            is_aromatic=False,
        )
        self._atom_type = atom_type

    @property
    def atomic_number(self):
        return self._atomic_number

    @property
    def atom_type(self):
        return self._atom_type

    @atom_type.setter
    def atom_type(self, potential):
        self._atom_type = potential


class System(BaseModel):
    """The OpenFF System object."""

    toolkit_forcefield: ToolkitForceField = None
    toolkit_topology: ToolkitTopology = None
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

    def populate_from_toolkit_data(self):
        """Construct a System from provided ForceField and Topology."""
        self.box = [10, 10, 10] * u.nm

        self.positions = np.random.random((self.toolkit_topology.n_topology_atoms, 3)) * u.nm

        self.topology = Topology(self.toolkit_topology)

        self.forcefield = dict()

        # Only doing on vdW for now
        matches = self.toolkit_forcefield.get_parameter_handler('vdW').find_matches(self.toolkit_topology)


        lj_map = {}

        for atom_key, atom_match in matches.items():
            atom_idx = atom_key[0]

            lj_type = atom_match.parameter_type

            if lj_type.sigma is None:
                sigma = 2. * lj_type.rmin_half / (2.**(1. / 6.))
            else:
                sigma = lj_type.sigma
            sigma = simtk_to_pint(sigma)
            epsilon = simtk_to_pint(lj_type.epsilon)

            potential = Potential(
                name=lj_type.id,
                expression='4*epsilon*((sigma/r)**12-(sigma/r)**6)',
                independent_variables={'r'},
                parameters={'sigma': sigma, 'epsilon': epsilon},
            )

            lj_map[atom_idx] = potential.name

            self.forcefield[potential.name] = potential

        for key, val in lj_map.items():
            self.topology.atoms[key].atom_type = self.forcefield[val]


    def from_file(self):
        raise NotImplementedError()

    def to_parmed(self):
        raise NotImplementedError()

    def to_openmm(self):
        raise NotImplementedError()
