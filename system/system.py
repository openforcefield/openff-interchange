from typing import Union, Iterable, List, Dict

import numpy as np
from pydantic import BaseModel, validator
import pint

from openforcefield.typing.engines.smirnoff import ForceField as ToolkitForceField
from openforcefield.topology import Topology as ToolkitTopology
from openforcefield.topology.molecule import Atom as ToolkitAtom

from .potential import ParametrizedAnalyticalPotential as Potential
from .utils import simtk_to_pint


u = pint.UnitRegistry()


class PotentialCollection(BaseModel):

    parameters: Dict[str, Potential]

    @classmethod
    def from_toolkit_forcefield(cls, toolkit_forcefield):

        for param in toolkit_forcefield.get_parameter_handler('vdW').parameters:
            if param.sigma is None:
                sigma = 2. * param.rmin_half / (2.**(1. / 6.))
            else:
                sigma = param.sigma
            sigma = simtk_to_pint(sigma)
            epsilon = simtk_to_pint(param.epsilon)

            potential = Potential(
                name=param.id,
                expression='4*epsilon*((sigma/r)**12-(sigma/r)**6)',
                independent_variables={'r'},
                parameters={'sigma': sigma, 'epsilon': epsilon},
            )

            try:
                forcefield.parameters[param.id] = potential
            except UnboundLocalError:
                forcefield = cls(parameters={param.id: potential})

        return forcefield


class Atom(ToolkitAtom):

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, atomic_number, parameter_id=None):
        super().__init__(
            atomic_number=atomic_number,
            formal_charge=0,
            is_aromatic=False,
        )
        self._parameter_id = parameter_id

    @property
    def atomic_number(self):
        return self._atomic_number

    @property
    def parameter_id(self):
        return self._parameter_id

    @parameter_id.setter
    def parameter_id(self, potential):
        self._parameter_id = potential


class Topology(BaseModel):

    atoms: List[ToolkitAtom]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_toolkit_topology(cls, toolkit_topology):

        atoms = []

        for atom in toolkit_topology.topology_atoms:
            atoms.append(atom.atom)

        return cls(atoms=atoms)


class System(BaseModel):
    """The OpenFF System object."""

    topology: Union[Topology, ToolkitTopology]
    forcefield: Union[PotentialCollection, ToolkitForceField]
    positions: Iterable = None
    box: Iterable = None

    @validator("forcefield")
    def validate_forcefield(cls, val):
        if isinstance(val, ToolkitForceField):
            return PotentialCollection.from_toolkit_forcefield(val)
        elif isinstance(val, PotentialCollection):
            return val
        else:
            raise TypeError

    @validator("topology")
    def validate_topology(cls, val):
        if isinstance(val, ToolkitTopology):
            return Topology.from_toolkit_topology(val)
        elif isinstance(val, Topology):
            return val
        else:
            raise TypeError

    @validator("*")
    def dummy_validator(cls, val):
        return val

    class Config:
        arbitrary_types_allowed = True

    def run_typing(self, toolkit_forcefield, toolkit_topology):
        # Only doing on vdW for now
        matches = toolkit_forcefield.get_parameter_handler('vdW').find_matches(toolkit_topology)

        typing_map = {}

        for atom_key, atom_match in matches.items():
            typing_map[atom_key[0]] = atom_match.parameter_type.id
            self.topology.atoms[atom_key[0]].parameter_id = atom_match.parameter_type.id

    def to_file(self):
        raise NotImplementedError()

    def from_file(self):
        raise NotImplementedError()

    def to_parmed(self):
        raise NotImplementedError()

    def to_openmm(self):
        raise NotImplementedError()
